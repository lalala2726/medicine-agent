import asyncio
from pathlib import Path

import pytest

from app.core.exception.exceptions import ServiceException
from app.core.mq.models import KnowledgeImportMessage
from app.rag.chunking import ChunkStrategyType
from app.rag.file_loader.types import FileKind, ParsedDocument
from app.services import knowledge_base_service


def test_submit_import_to_queue_publishes_one_message_per_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    测试目的：验证导入提交会按 file_urls 数量拆分并发布多条 MQ 消息。
    预期结果：publish_import_messages 收到与 URL 数一致的消息，且返回 accepted_count 正确。
    """
    captured_messages: list[KnowledgeImportMessage] = []

    async def _fake_publish(messages: list[KnowledgeImportMessage]) -> None:
        captured_messages.extend(messages)

    monkeypatch.setattr(
        knowledge_base_service,
        "publish_import_messages",
        _fake_publish,
    )

    result = asyncio.run(
        knowledge_base_service.submit_import_to_queue(
            knowledge_name="demo",
            document_id=10,
            file_url=[
                "https://example.com/a.txt",
                "https://example.com/b.txt",
            ],
            chunk_strategy=ChunkStrategyType.CHARACTER,
            chunk_size=200,
            token_size=50,
        )
    )

    assert result["accepted_count"] == 2
    assert len(result["task_uuids"]) == 2
    assert len(captured_messages) == 2
    assert captured_messages[0].file_url == "https://example.com/a.txt"
    assert captured_messages[1].file_url == "https://example.com/b.txt"


def test_import_knowledge_service_rejects_url_without_supported_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    测试目的：验证导入流程会在下载前执行 URL 后缀校验。
    预期结果：不支持后缀的 URL 被加入 failed_urls，且不会触发下载。
    """
    called = {"download": False}

    def _fake_download(_url: str):
        called["download"] = True
        return "a.txt", Path("/tmp/a.txt")

    monkeypatch.setattr(knowledge_base_service, "_download_file", _fake_download)
    monkeypatch.setattr(knowledge_base_service, "_print_chunks_to_console", lambda **_: None)

    result = knowledge_base_service.import_knowledge_service(
        knowledge_name="demo",
        document_id=1,
        file_url=["https://example.com/file.bin"],
        chunk_strategy=ChunkStrategyType.CHARACTER,
        chunk_size=200,
        token_size=50,
    )

    assert result["results"] == []
    assert result["failed_urls"] == ["https://example.com/file.bin"]
    assert called["download"] is False


def test_import_knowledge_service_runs_download_parse_and_chunk(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    测试目的：验证导入流程按“下载 -> 解析 -> 切片”主链路运行并返回文本直出结构。
    预期结果：results 含 1 条成功记录，且 chunk_count 大于 0 且 text_length 大于 0。
    """
    source_path = tmp_path / "demo.txt"
    source_path.write_text("dummy", encoding="utf-8")

    monkeypatch.setattr(
        knowledge_base_service,
        "_download_file",
        lambda _url: ("demo.txt", source_path),
    )
    monkeypatch.setattr(
        knowledge_base_service,
        "parse_downloaded_file",
        lambda **_: ParsedDocument(
            file_kind=FileKind.TEXT,
            mime_type="text/plain",
            source_extension=".txt",
            text="第一段\n\n第二段",
        ),
    )
    monkeypatch.setattr(knowledge_base_service, "_print_chunks_to_console", lambda **_: None)

    result = knowledge_base_service.import_knowledge_service(
        knowledge_name="demo",
        document_id=2,
        file_url=["https://example.com/demo.txt"],
        chunk_strategy=ChunkStrategyType.CHARACTER,
        chunk_size=200,
        token_size=50,
    )

    assert result["failed_urls"] == []
    assert len(result["results"]) == 1
    first = result["results"][0]
    assert first["filename"] == "demo.txt"
    assert first["file_kind"] == "text"
    assert first["mime_type"] == "text/plain"
    assert first["source_extension"] == ".txt"
    assert first["text_length"] > 0
    assert first["chunk_count"] > 0
    assert "chunks" in first


def test_import_knowledge_service_keeps_downloaded_file_on_parse_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    测试目的：验证解析失败时不会删除已下载源文件，便于后续排障。
    预期结果：failed_urls 包含失败 URL，且本地下载文件仍存在。
    """
    source_path = tmp_path / "demo.txt"
    source_path.write_text("dummy", encoding="utf-8")

    monkeypatch.setattr(
        knowledge_base_service,
        "_download_file",
        lambda _url: ("demo.txt", source_path),
    )
    monkeypatch.setattr(
        knowledge_base_service,
        "parse_downloaded_file",
        lambda **_: (_raise_parse_error()),
    )
    monkeypatch.setattr(knowledge_base_service, "_print_chunks_to_console", lambda **_: None)

    result = knowledge_base_service.import_knowledge_service(
        knowledge_name="demo",
        document_id=3,
        file_url=["https://example.com/demo.txt"],
        chunk_strategy=ChunkStrategyType.CHARACTER,
        chunk_size=10,
        token_size=20,
    )

    assert result["results"] == []
    assert result["failed_urls"] == ["https://example.com/demo.txt"]
    assert source_path.exists()


def test_import_knowledge_service_does_not_trigger_vectorization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    测试目的：验证本期导入链路只做下载/解析/切片打印，不执行向量化与向量写入。
    预期结果：导入成功时 embed_texts/insert_embeddings 均不会被调用。
    """
    source_path = tmp_path / "demo.txt"
    source_path.write_text("dummy", encoding="utf-8")
    called = {"embed": False, "insert": False}

    def _fake_embed(_texts: list[str]) -> list[list[float]]:
        called["embed"] = True
        return [[0.1, 0.2]]

    def _fake_insert(*_args, **_kwargs) -> None:
        called["insert"] = True

    monkeypatch.setattr(
        knowledge_base_service,
        "_download_file",
        lambda _url: ("demo.txt", source_path),
    )
    monkeypatch.setattr(
        knowledge_base_service,
        "parse_downloaded_file",
        lambda **_: ParsedDocument(
            file_kind=FileKind.TEXT,
            mime_type="text/plain",
            source_extension=".txt",
            text="只做切片，不做向量化",
        ),
    )
    monkeypatch.setattr(knowledge_base_service, "embed_texts", _fake_embed)
    monkeypatch.setattr(
        knowledge_base_service.vector_repository,
        "insert_embeddings",
        _fake_insert,
    )
    monkeypatch.setattr(knowledge_base_service, "_print_chunks_to_console", lambda **_: None)

    result = knowledge_base_service.import_knowledge_service(
        knowledge_name="demo",
        document_id=5,
        file_url=["https://example.com/demo.txt"],
        chunk_strategy=ChunkStrategyType.CHARACTER,
        chunk_size=100,
        token_size=50,
    )

    assert result["failed_urls"] == []
    assert called["embed"] is False
    assert called["insert"] is False


def _raise_parse_error() -> ParsedDocument:
    """
    测试目的：构造解析失败场景，验证导入流程的失败分支行为。
    预期结果：调用时抛出 ServiceException。
    """
    raise ServiceException("mock parse error")
