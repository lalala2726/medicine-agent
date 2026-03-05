import pytest

from app.core.exception.exceptions import ServiceException
from app.core.mq.config.settings import get_rabbitmq_settings


def test_get_rabbitmq_settings_loads_defaults(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证可选环境变量缺失时会加载默认值。"""
    monkeypatch.setenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
    monkeypatch.delenv("RABBITMQ_EXCHANGE", raising=False)
    monkeypatch.delenv("RABBITMQ_COMMAND_QUEUE", raising=False)
    monkeypatch.delenv("RABBITMQ_COMMAND_ROUTING_KEY", raising=False)
    monkeypatch.delenv("RABBITMQ_RESULT_ROUTING_KEY", raising=False)
    monkeypatch.delenv("RABBITMQ_PREFETCH_COUNT", raising=False)
    monkeypatch.delenv("KNOWLEDGE_LATEST_VERSION_KEY_PREFIX", raising=False)

    get_rabbitmq_settings.cache_clear()
    settings = get_rabbitmq_settings()

    assert settings.exchange == "knowledge.import"
    assert settings.command_queue == "knowledge.import.command.q"
    assert settings.command_routing_key == "knowledge.import.command"
    assert settings.result_routing_key == "knowledge.import.result"
    assert settings.prefetch_count == 1
    assert settings.latest_version_key_prefix == "kb:latest"

    get_rabbitmq_settings.cache_clear()


def test_get_rabbitmq_settings_requires_rabbitmq_url(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证缺少 RabbitMQ URL 时会抛出配置异常。"""
    monkeypatch.delenv("RABBITMQ_URL", raising=False)
    get_rabbitmq_settings.cache_clear()

    with pytest.raises(ServiceException):
        get_rabbitmq_settings()

    get_rabbitmq_settings.cache_clear()


def test_get_rabbitmq_settings_rejects_non_positive_prefetch(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证非法 prefetch 配置会被拒绝。"""
    monkeypatch.setenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
    monkeypatch.setenv("RABBITMQ_PREFETCH_COUNT", "0")
    get_rabbitmq_settings.cache_clear()

    with pytest.raises(ServiceException):
        get_rabbitmq_settings()

    get_rabbitmq_settings.cache_clear()
