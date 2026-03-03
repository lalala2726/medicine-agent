from pathlib import Path

import pytest

from app.core.exception.exceptions import ServiceException
from app.rag.chunking import ChunkStrategyType
from app.rag.file_loader.types import FileKind, ParsedDocument, ParsedPage
from app.services import knowledge_base_service


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
    测试目的：验证导入流程按“下载 -> 解析 -> 切片”主链路运行并返回解析信息。
    预期结果：results 含 1 条成功记录，且 chunk_count 大于 0。
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
            pages=[ParsedPage(page_number=1, text="第一段\n\n第二段")],
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
    assert first["chunk_count"] > 0


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


def _raise_parse_error() -> ParsedDocument:
    """
    测试目的：构造解析失败场景，验证导入流程的失败分支行为。
    预期结果：调用时抛出 ServiceException。
    """
    raise ServiceException("mock parse error")
