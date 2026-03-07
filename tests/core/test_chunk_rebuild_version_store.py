import pytest

from app.core.mq.config.document.chunk_rebuild_settings import (
    ChunkRebuildRabbitMQSettings,
)
from app.core.mq.state.document.chunk_rebuild_version_store import (
    build_version_key,
    get_latest_version,
    is_stale,
)


class _FakeRedis:
    def __init__(self, values: dict[str, object] | None = None) -> None:
        self.values = values or {}

    def get(self, key: str):
        return self.values.get(key)


def _build_settings() -> ChunkRebuildRabbitMQSettings:
    return ChunkRebuildRabbitMQSettings(
        exchange="knowledge.chunk_rebuild",
        command_queue="knowledge.chunk_rebuild.command.q",
        command_routing_key="knowledge.chunk_rebuild.command",
        result_routing_key="knowledge.chunk_rebuild.result",
        prefetch_count=1,
        latest_version_key_prefix="kb:chunk_edit:latest_version",
    )


def test_build_version_key_uses_vector_id() -> None:
    key = build_version_key(vector_id=101, settings=_build_settings())
    assert key == "kb:chunk_edit:latest_version:101"


def test_get_latest_version_reads_int(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.core.mq.state._version_store_support.get_redis_connection",
        lambda: _FakeRedis({"kb:chunk_edit:latest_version:101": b"6"}),
    )

    latest_version = get_latest_version(vector_id=101, settings=_build_settings())

    assert latest_version == 6


def test_is_stale_returns_true_for_older_version(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.core.mq.state._version_store_support.get_redis_connection",
        lambda: _FakeRedis({"kb:chunk_edit:latest_version:101": b"6"}),
    )

    assert is_stale(
        vector_id=101,
        version=5,
        settings=_build_settings(),
    ) is True
