import pytest

from app.core.exception.exceptions import ServiceException
from app.core.mq.config.document.chunk_rebuild_settings import (
    get_chunk_rebuild_settings,
)
from app.core.mq.config.common import is_chunk_rebuild_consumer_enabled


def test_get_chunk_rebuild_settings_loads_defaults(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证切片重建 MQ 可选环境变量缺失时会加载默认值。"""
    monkeypatch.setenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
    monkeypatch.delenv("RABBITMQ_CHUNK_REBUILD_EXCHANGE", raising=False)
    monkeypatch.delenv("RABBITMQ_CHUNK_REBUILD_COMMAND_QUEUE", raising=False)
    monkeypatch.delenv("RABBITMQ_CHUNK_REBUILD_COMMAND_ROUTING_KEY", raising=False)
    monkeypatch.delenv("RABBITMQ_CHUNK_REBUILD_RESULT_ROUTING_KEY", raising=False)
    monkeypatch.delenv("KNOWLEDGE_CHUNK_EDIT_LATEST_VERSION_KEY_PREFIX", raising=False)
    monkeypatch.delenv("RABBITMQ_PREFETCH_COUNT", raising=False)

    get_chunk_rebuild_settings.cache_clear()
    settings = get_chunk_rebuild_settings()

    assert settings.exchange == "knowledge.chunk_rebuild"
    assert settings.command_queue == "knowledge.chunk_rebuild.command.q"
    assert settings.command_routing_key == "knowledge.chunk_rebuild.command"
    assert settings.result_routing_key == "knowledge.chunk_rebuild.result"
    assert settings.prefetch_count == 1
    assert settings.latest_version_key_prefix == "kb:chunk_edit:latest_version"

    get_chunk_rebuild_settings.cache_clear()


def test_get_chunk_rebuild_settings_requires_rabbitmq_url(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证缺少 RabbitMQ URL 时会抛出切片重建 MQ 配置异常。"""
    monkeypatch.delenv("RABBITMQ_URL", raising=False)
    get_chunk_rebuild_settings.cache_clear()

    with pytest.raises(ServiceException):
        get_chunk_rebuild_settings()

    get_chunk_rebuild_settings.cache_clear()


def test_is_chunk_rebuild_consumer_enabled_defaults_true(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证切片重建消费者开关默认开启。"""
    monkeypatch.delenv("MQ_CHUNK_REBUILD_CONSUMER_ENABLED", raising=False)

    assert is_chunk_rebuild_consumer_enabled() is True


def test_is_chunk_rebuild_consumer_enabled_supports_false(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证切片重建消费者开关可显式关闭。"""
    monkeypatch.setenv("MQ_CHUNK_REBUILD_CONSUMER_ENABLED", "false")

    assert is_chunk_rebuild_consumer_enabled() is False
