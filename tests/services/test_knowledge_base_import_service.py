import asyncio
from pathlib import Path

import pytest

from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException
from app.core.mq.models import KnowledgeImportMessage
from app.rag.chunking import ChunkStats, ChunkStrategyType, SplitChunk
from app.rag.file_loader.types import FileKind, ParsedDocument
from app.services import knowledge_base_service


def test_submit_import_to_queue_publishes_one_message_per_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    测试目的：验证导入提交会按 file_urls 数量拆分并发布多条 MQ 消息，并透传 embedding_model。
    预期结果：publish_import_messages 收到与 URL 数一致的消息，且每条消息包含 embedding_model。
    """
    captured_messages: list[KnowledgeImportMessage] = []

    async def _fake_publish(messages: list[KnowledgeImportMessage]) -> None:
        captured_messages.extend(messages)

    monkeypatch.setattr(
        knowledge_base_service,
        "publish_import_messages",
        _fake_publish,
    )
    monkeypatch.setattr(
        knowledge_base_service.vector_repository,
        "ensure_collection_exists",
        lambda **_: None,
    )

    result = asyncio.run(
        knowledge_base_service.submit_import_to_queue(
            knowledge_name="demo",
            document_id=10,
            file_url=[
                "https://example.com/a.txt",
                "https://example.com/b.txt",
            ],
            embedding_model="text-embedding-v4",
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
    assert captured_messages[0].embedding_model == "text-embedding-v4"


def test_submit_import_to_queue_rejects_unknown_collection_before_publish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    测试目的：验证入队前会先校验知识库集合存在，不存在时直接拒绝且不发布 MQ 消息。
    预期结果：submit_import_to_queue 抛出 NOT_FOUND 业务异常，publish_import_messages 不会被调用。
    """
    called = {"publish": 0}

    async def _fake_publish(_messages: list[KnowledgeImportMessage]) -> None:
        called["publish"] += 1

    def _raise_not_found(**_kwargs) -> None:
        raise ServiceException(
            code=ResponseCode.NOT_FOUND,
            message="知识库不存在",
        )

    monkeypatch.setattr(
        knowledge_base_service,
        "publish_import_messages",
        _fake_publish,
    )
    monkeypatch.setattr(
        knowledge_base_service.vector_repository,
        "ensure_collection_exists",
        _raise_not_found,
    )

    with pytest.raises(ServiceException) as exc_info:
        asyncio.run(
            knowledge_base_service.submit_import_to_queue(
                knowledge_name="not-exists",
                document_id=10,
                file_url=["https://example.com/a.txt"],
                embedding_model="text-embedding-v4",
                chunk_strategy=ChunkStrategyType.CHARACTER,
                chunk_size=200,
                token_size=50,
            )
        )

    assert int(exc_info.value.code) == ResponseCode.NOT_FOUND.code
    assert str(exc_info.value) == "知识库不存在，无法提交导入任务"
    assert called["publish"] == 0


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
    monkeypatch.setattr(
        knowledge_base_service.vector_repository,
        "ensure_collection_exists",
        lambda **_: None,
    )
    monkeypatch.setattr(
        knowledge_base_service.vector_repository,
        "get_collection_embedding_dim",
        lambda **_: 1024,
    )
    monkeypatch.setattr(
        knowledge_base_service,
        "_create_embedding_client",
        lambda **_: object(),
    )

    result = knowledge_base_service.import_knowledge_service(
        knowledge_name="demo",
        document_id=1,
        file_url=["https://example.com/file.bin"],
        embedding_model="text-embedding-v4",
        chunk_strategy=ChunkStrategyType.CHARACTER,
        chunk_size=200,
        token_size=50,
    )

    assert result["results"] == []
    assert result["failed_urls"] == ["https://example.com/file.bin"]
    assert called["download"] is False


def test_import_knowledge_service_runs_vectorization_and_insert_batches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    测试目的：验证导入流程按“下载 -> 解析 -> 切片 -> 向量化 -> 入库”主链路运行。
    预期结果：results 含 1 条成功记录，返回 vector_count/insert_batches，且入库字段包含关键快照信息。
    """
    source_path = tmp_path / "demo.txt"
    source_path.write_text("dummy", encoding="utf-8")
    insert_calls: list[dict] = []

    monkeypatch.setenv("KNOWLEDGE_VECTOR_BATCH_SIZE", "2")
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
            text="第一段\n\n第二段\n\n第三段",
        ),
    )
    monkeypatch.setattr(
        knowledge_base_service.vector_repository,
        "ensure_collection_exists",
        lambda **_: None,
    )
    monkeypatch.setattr(
        knowledge_base_service.vector_repository,
        "get_collection_embedding_dim",
        lambda **_: 1024,
    )
    monkeypatch.setattr(
        knowledge_base_service,
        "_create_embedding_client",
        lambda **_: object(),
    )
    monkeypatch.setattr(
        knowledge_base_service,
        "embed_texts",
        lambda texts, **_: [[0.1, 0.2] for _ in texts],
    )
    monkeypatch.setattr(
        knowledge_base_service.vector_repository,
        "insert_embeddings",
        lambda **kwargs: insert_calls.append(kwargs),
    )
    monkeypatch.setattr(
        knowledge_base_service,
        "_split_parsed_text",
        lambda *_args, **_kwargs: [
            SplitChunk(text="A", stats=ChunkStats(char_count=1)),
            SplitChunk(text="B", stats=ChunkStats(char_count=1)),
            SplitChunk(text="C", stats=ChunkStats(char_count=1)),
        ],
    )
    monkeypatch.setattr(knowledge_base_service, "_print_chunks_to_console", lambda **_: None)

    result = knowledge_base_service.import_knowledge_service(
        knowledge_name="demo",
        document_id=2,
        file_url=["https://example.com/demo.txt"],
        embedding_model="text-embedding-v4",
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
    assert first["chunk_count"] == 3
    assert first["vector_count"] == 3
    assert first["insert_batches"] == 2
    assert first["embedding_model"] == "text-embedding-v4"
    assert first["embedding_dim"] == 1024
    assert first["callback_status"] == "PENDING"
    assert len(insert_calls) == 2
    assert insert_calls[0]["start_chunk_no"] == 1
    assert insert_calls[1]["start_chunk_no"] == 3
    assert insert_calls[0]["chunk_strategy"] == "character"
    assert insert_calls[0]["chunk_size"] == 200
    assert insert_calls[0]["token_size"] == 50


def test_import_knowledge_service_keeps_downloaded_file_on_parse_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    测试目的：验证解析失败时不会删除已下载源文件，便于后续排障。
    预期结果：failed_urls 包含失败 URL，且本地下载文件仍存在，同时失败明细包含错误信息。
    """
    source_path = tmp_path / "demo.txt"
    source_path.write_text("dummy", encoding="utf-8")

    monkeypatch.setattr(
        knowledge_base_service,
        "_download_file",
        lambda _url: ("demo.txt", source_path),
    )
    monkeypatch.setattr(
        knowledge_base_service.vector_repository,
        "ensure_collection_exists",
        lambda **_: None,
    )
    monkeypatch.setattr(
        knowledge_base_service.vector_repository,
        "get_collection_embedding_dim",
        lambda **_: 1024,
    )
    monkeypatch.setattr(
        knowledge_base_service,
        "_create_embedding_client",
        lambda **_: object(),
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
        embedding_model="text-embedding-v4",
        chunk_strategy=ChunkStrategyType.CHARACTER,
        chunk_size=10,
        token_size=20,
    )

    assert result["results"] == []
    assert result["failed_urls"] == ["https://example.com/demo.txt"]
    assert len(result["failed_details"]) == 1
    assert "mock parse error" in result["failed_details"][0]["error"]
    assert source_path.exists()


def test_import_knowledge_service_batches_are_strictly_serial(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    测试目的：验证向量化与入库按批次严格串行执行（先向量化该批，再入库该批）。
    预期结果：执行轨迹满足 embed-1 -> insert-1 -> embed-2 -> insert-2。
    """
    source_path = tmp_path / "demo.txt"
    source_path.write_text("dummy", encoding="utf-8")
    trace: list[str] = []

    monkeypatch.setenv("KNOWLEDGE_VECTOR_BATCH_SIZE", "2")
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
            text="T",
        ),
    )
    monkeypatch.setattr(
        knowledge_base_service.vector_repository,
        "ensure_collection_exists",
        lambda **_: None,
    )
    monkeypatch.setattr(
        knowledge_base_service.vector_repository,
        "get_collection_embedding_dim",
        lambda **_: 1024,
    )
    monkeypatch.setattr(
        knowledge_base_service,
        "_create_embedding_client",
        lambda **_: object(),
    )

    def _fake_embed(texts: list[str], **_kwargs) -> list[list[float]]:
        trace.append(f"embed-{len(texts)}")
        return [[0.1, 0.2] for _ in texts]

    def _fake_insert(**kwargs) -> None:
        trace.append(f"insert-{len(kwargs['texts'])}")

    monkeypatch.setattr(knowledge_base_service, "embed_texts", _fake_embed)
    monkeypatch.setattr(
        knowledge_base_service.vector_repository,
        "insert_embeddings",
        _fake_insert,
    )
    monkeypatch.setattr(
        knowledge_base_service,
        "_split_parsed_text",
        lambda *_args, **_kwargs: [
            SplitChunk(text="A", stats=ChunkStats(char_count=1)),
            SplitChunk(text="B", stats=ChunkStats(char_count=1)),
            SplitChunk(text="C", stats=ChunkStats(char_count=1)),
        ],
    )
    monkeypatch.setattr(knowledge_base_service, "_print_chunks_to_console", lambda **_: None)

    result = knowledge_base_service.import_knowledge_service(
        knowledge_name="demo",
        document_id=6,
        file_url=["https://example.com/demo.txt"],
        embedding_model="text-embedding-v4",
        chunk_strategy=ChunkStrategyType.CHARACTER,
        chunk_size=100,
        token_size=50,
    )

    assert result["failed_urls"] == []
    assert trace == ["embed-2", "insert-2", "embed-1", "insert-1"]


def _raise_parse_error() -> ParsedDocument:
    """
    测试目的：构造解析失败场景，验证导入流程的失败分支行为。
    预期结果：调用时抛出 ServiceException。
    """
    raise ServiceException("mock parse error")
