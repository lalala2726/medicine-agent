import pytest

import app.core.mq.config.common as mq_common
from app.core.mq.config.document.chunk_rebuild_settings import (
    CHUNK_EDIT_LATEST_VERSION_KEY_PREFIX,
    get_chunk_rebuild_settings,
)


def test_get_chunk_rebuild_settings_loads_defaults(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证切片重建 MQ 配置固定使用代码常量，不受环境变量影响。"""
    monkeypatch.delenv("RABBITMQ_URL", raising=False)
    monkeypatch.setenv("RABBITMQ_CHUNK_REBUILD_EXCHANGE", "unexpected.exchange")
    monkeypatch.setenv("RABBITMQ_CHUNK_REBUILD_COMMAND_QUEUE", "unexpected.queue")
    monkeypatch.setenv("RABBITMQ_CHUNK_REBUILD_COMMAND_ROUTING_KEY", "unexpected.command")
    monkeypatch.setenv("RABBITMQ_CHUNK_REBUILD_RESULT_ROUTING_KEY", "unexpected.result")
    monkeypatch.setenv("KNOWLEDGE_CHUNK_EDIT_LATEST_VERSION_KEY_PREFIX", "unexpected:prefix")
    monkeypatch.setenv("RABBITMQ_PREFETCH_COUNT", "99")

    get_chunk_rebuild_settings.cache_clear()
    settings = get_chunk_rebuild_settings()

    assert settings.exchange == "knowledge.chunk_rebuild"
    assert settings.command_queue == "knowledge.chunk_rebuild.command.q"
    assert settings.command_routing_key == "knowledge.chunk_rebuild.command"
    assert settings.result_routing_key == "knowledge.chunk_rebuild.result"
    assert settings.prefetch_count == 1
    assert settings.latest_version_key_prefix == CHUNK_EDIT_LATEST_VERSION_KEY_PREFIX

    get_chunk_rebuild_settings.cache_clear()


def test_is_chunk_rebuild_consumer_enabled_defaults_true(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证切片重建消费者开关默认开启。"""
    monkeypatch.setattr(mq_common, "CHUNK_REBUILD_CONSUMER_ENABLED", True)

    assert mq_common.is_chunk_rebuild_consumer_enabled() is True


def test_is_chunk_rebuild_consumer_enabled_supports_false(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证切片重建消费者开关可通过代码常量关闭。"""
    monkeypatch.setattr(mq_common, "CHUNK_REBUILD_CONSUMER_ENABLED", False)

    assert mq_common.is_chunk_rebuild_consumer_enabled() is False
