import pytest

from app.core.mq.config.document.import_settings import (
    IMPORT_LATEST_VERSION_KEY_PREFIX,
    get_import_settings,
)


def test_get_import_settings_loads_defaults(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证导入链路配置固定使用代码常量，不受环境变量影响。"""
    monkeypatch.delenv("RABBITMQ_URL", raising=False)
    monkeypatch.setenv("RABBITMQ_EXCHANGE", "unexpected.exchange")
    monkeypatch.setenv("RABBITMQ_COMMAND_QUEUE", "unexpected.queue")
    monkeypatch.setenv("RABBITMQ_COMMAND_ROUTING_KEY", "unexpected.command")
    monkeypatch.setenv("RABBITMQ_RESULT_ROUTING_KEY", "unexpected.result")
    monkeypatch.setenv("RABBITMQ_PREFETCH_COUNT", "99")
    monkeypatch.setenv("KNOWLEDGE_LATEST_VERSION_KEY_PREFIX", "unexpected:prefix")

    get_import_settings.cache_clear()
    settings = get_import_settings()

    assert settings.exchange == "knowledge.import"
    assert settings.command_queue == "knowledge.import.command.q"
    assert settings.command_routing_key == "knowledge.import.command"
    assert settings.result_routing_key == "knowledge.import.result"
    assert settings.prefetch_count == 1
    assert settings.latest_version_key_prefix == IMPORT_LATEST_VERSION_KEY_PREFIX

    get_import_settings.cache_clear()
