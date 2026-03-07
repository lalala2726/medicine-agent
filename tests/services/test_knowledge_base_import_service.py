from pathlib import Path

import pytest

from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException
from app.core.mq.contracts.import_models import ProcessingStageDetail
from app.rag.chunking import ChunkStats, ChunkStrategyType, SplitChunk
from app.rag.file_loader.types import FileKind, ParsedDocument
from app.services import knowledge_base_service


def test_import_single_file_emits_processing_stage_callbacks(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
) -> None:
    """验证 import_single_file 会按预期顺序触发处理阶段回调。"""
    source_path = tmp_path / "demo.txt"
    source_path.write_text("dummy", encoding="utf-8")
    observed: list[ProcessingStageDetail] = []

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
        "create_embedding_client",
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
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        knowledge_base_service.vector_repository,
        "count_document_chunks",
        lambda **_kwargs: 3,
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

    result = knowledge_base_service.import_single_file(
        url="https://example.com/demo.txt",
        knowledge_name="demo",
        document_id=2,
        embedding_model="text-embedding-v4",
        chunk_strategy=ChunkStrategyType.CHARACTER,
        chunk_size=200,
        token_size=50,
        on_processing_stage=observed.append,
    )

    assert result.status == "success"
    assert observed == [
        ProcessingStageDetail.DOWNLOADING,
        ProcessingStageDetail.PARSING,
        ProcessingStageDetail.CHUNKING,
        ProcessingStageDetail.EMBEDDING,
        ProcessingStageDetail.INSERTING,
    ]


def test_import_single_file_rejects_url_without_supported_suffix(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证不支持的 URL 后缀会在下载前被拒绝。"""
    called = {"download": False}

    def _fake_download(_url: str):
        called["download"] = True
        return "a.txt", Path("/tmp/a.txt")

    monkeypatch.setattr(knowledge_base_service, "_download_file", _fake_download)
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
        "create_embedding_client",
        lambda **_: object(),
    )

    result = knowledge_base_service.import_single_file(
        url="https://example.com/file.bin",
        knowledge_name="demo",
        document_id=1,
        embedding_model="text-embedding-v4",
        chunk_strategy=ChunkStrategyType.CHARACTER,
        chunk_size=200,
        token_size=50,
    )

    assert result.status == "failed"
    assert result.file_url == "https://example.com/file.bin"
    assert called["download"] is False


def test_import_single_file_runs_vectorization_and_insert_batches(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
) -> None:
    """验证导入主流程可完成下载、解析、切片、向量化与入库。"""
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
        "create_embedding_client",
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
        knowledge_base_service.vector_repository,
        "count_document_chunks",
        lambda **_kwargs: 3,
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

    result = knowledge_base_service.import_single_file(
        url="https://example.com/demo.txt",
        knowledge_name="demo",
        document_id=2,
        embedding_model="text-embedding-v4",
        chunk_strategy=ChunkStrategyType.CHARACTER,
        chunk_size=200,
        token_size=50,
    )

    assert result.status == "success"
    first = result
    assert first.filename == "demo.txt"
    assert first.file_kind == "text"
    assert first.mime_type == "text/plain"
    assert first.source_extension == ".txt"
    assert first.chunk_count == 3
    assert first.vector_count == 3
    assert first.insert_batches == 2
    assert first.embedding_model == "text-embedding-v4"
    assert first.embedding_dim == 1024
    assert len(insert_calls) == 2
    assert insert_calls[0]["start_chunk_index"] == 1
    assert insert_calls[1]["start_chunk_index"] == 3
    assert insert_calls[0]["chunk_strategy"] == "character"
    assert insert_calls[0]["chunk_size"] == 200
    assert insert_calls[0]["token_size"] == 50


def test_import_single_file_keeps_downloaded_file_on_parse_failure(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
) -> None:
    """验证解析失败时保留已下载源文件。"""
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
        "create_embedding_client",
        lambda **_: object(),
    )
    monkeypatch.setattr(
        knowledge_base_service,
        "parse_downloaded_file",
        lambda **_: (_raise_parse_error()),
    )

    result = knowledge_base_service.import_single_file(
        url="https://example.com/demo.txt",
        knowledge_name="demo",
        document_id=3,
        embedding_model="text-embedding-v4",
        chunk_strategy=ChunkStrategyType.CHARACTER,
        chunk_size=10,
        token_size=20,
    )

    assert result.status == "failed"
    assert result.file_url == "https://example.com/demo.txt"
    assert "mock parse error" in result.error
    assert source_path.exists()


def test_import_single_file_batches_are_strictly_serial(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
) -> None:
    """验证向量化与入库按批次严格串行执行。"""
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
        "create_embedding_client",
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
        knowledge_base_service.vector_repository,
        "count_document_chunks",
        lambda **_kwargs: 3,
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

    result = knowledge_base_service.import_single_file(
        url="https://example.com/demo.txt",
        knowledge_name="demo",
        document_id=6,
        embedding_model="text-embedding-v4",
        chunk_strategy=ChunkStrategyType.CHARACTER,
        chunk_size=100,
        token_size=50,
    )

    assert result.status == "success"
    assert trace == ["embed-2", "insert-2", "embed-1", "insert-1"]


def test_import_single_file_fails_when_insert_visibility_check_not_passed(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
) -> None:
    """验证写入可见性校验失败时返回 failed，避免误发 completed。"""
    source_path = tmp_path / "demo.txt"
    source_path.write_text("dummy", encoding="utf-8")

    monkeypatch.setenv("KNOWLEDGE_VECTOR_BATCH_SIZE", "2")
    monkeypatch.setattr(
        knowledge_base_service,
        "DEFAULT_INSERT_VERIFY_MAX_RETRIES",
        2,
    )
    monkeypatch.setattr(
        knowledge_base_service,
        "DEFAULT_INSERT_VERIFY_INTERVAL_SECONDS",
        0.0,
    )
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
        "create_embedding_client",
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
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        knowledge_base_service.vector_repository,
        "count_document_chunks",
        lambda **_kwargs: 0,
    )
    monkeypatch.setattr(
        knowledge_base_service,
        "_split_parsed_text",
        lambda *_args, **_kwargs: [
            SplitChunk(text="A", stats=ChunkStats(char_count=1)),
            SplitChunk(text="B", stats=ChunkStats(char_count=1)),
        ],
    )

    result = knowledge_base_service.import_single_file(
        url="https://example.com/demo.txt",
        knowledge_name="demo",
        document_id=9,
        embedding_model="text-embedding-v4",
        chunk_strategy=ChunkStrategyType.CHARACTER,
        chunk_size=200,
        token_size=50,
    )

    assert result.status == "failed"
    assert "切片写入校验失败" in result.error


def _raise_parse_error() -> ParsedDocument:
    """用于失败路径测试，抛出解析异常。"""
    raise ServiceException(code=ResponseCode.OPERATION_FAILED, message="mock parse error")
