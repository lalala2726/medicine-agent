from pathlib import Path

import pytest

from app.core.chunking import ChunkStrategyType
from app.services import knowledge_base_service


class DummyClient:
    def __init__(self) -> None:
        self.inserted_data: list[dict] = []

    def has_collection(self, name: str) -> bool:
        return True

    def describe_collection(self, name: str) -> dict:
        return {
            "fields": [
                {"name": "embedding", "params": {"dim": "3"}},
                {"name": "content"},
                {"name": "knowledge_base_id"},
            ]
        }

    def insert(self, collection_name: str, data: list[dict]) -> None:
        self.inserted_data.extend(data)
        return None


def test_import_knowledge_service_registers_and_cleanup(monkeypatch, tmp_path):
    # 伪造 milvus client
    dummy_client = DummyClient()
    monkeypatch.setattr(
        knowledge_base_service.vector_service,
        "get_milvus_client",
        lambda: dummy_client,
    )
    monkeypatch.setattr(
        knowledge_base_service.vector_service,
        "embed_texts",
        lambda texts: [[0.0, 0.0, 0.0] for _ in texts],
    )

    # 控制图片临时目录父路径
    monkeypatch.setenv("FILE_IMAGE_OUTPUT_DIR", str(tmp_path))

    created = {}

    def fake_download_file(url: str):
        file_path = tmp_path / "download.txt"
        file_path.write_text("data", encoding="utf-8")
        created["path"] = file_path
        return "download.txt", file_path

    monkeypatch.setattr(knowledge_base_service, "_download_file", fake_download_file)

    result = knowledge_base_service.import_knowledge_service(
        "demo",
        123,
        ["http://example.com/a.txt"],
        ChunkStrategyType.LENGTH,
        500,
        100,
    )

    assert result["failed_urls"] == []
    assert len(result["results"]) == 1
    parsed = result["results"][0]
    assert parsed["filename"] == "download.txt"
    assert parsed["pages"][0]["text"].strip() == "data"
    assert len(dummy_client.inserted_data) == 1
    assert dummy_client.inserted_data[0]["content"].strip() == "data"

    image_dir = Path(parsed["image_dir"])
    assert image_dir.exists()
    assert created["path"].exists()

    cleanup_result = knowledge_base_service.cleanup_import_assets("download.txt")
    assert cleanup_result["deleted_count"] == 1
    assert not image_dir.exists()
    assert not created["path"].exists()


def test_import_knowledge_service_raises_when_collection_missing(monkeypatch):
    class MissingClient:
        def has_collection(self, name: str) -> bool:
            return False

    monkeypatch.setattr(
        knowledge_base_service.vector_service,
        "get_milvus_client",
        lambda: MissingClient(),
    )

    with pytest.raises(knowledge_base_service.ServiceException):
        knowledge_base_service.import_knowledge_service(
            "missing",
            123,
            ["http://example.com/a.txt"],
            ChunkStrategyType.LENGTH,
            500,
            100,
        )
