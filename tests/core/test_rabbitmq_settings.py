import pytest

from app.core.exception.exceptions import ServiceException
from app.core.mq.settings import get_rabbitmq_settings


def test_get_rabbitmq_settings_supports_duration_suffixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    测试目的：验证 MQ_RETRY_DELAYS_SECONDS 支持 s/m/h 单位并可正确换算为秒。
    预期结果：配置解析后的 retry_delays_seconds 与预期秒数一致，且默认 max_retries 等于延迟项数量。
    """
    monkeypatch.setenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
    monkeypatch.setenv(
        "MQ_RETRY_DELAYS_SECONDS",
        "15s,15s,30s,3m,10m,20m,30m,30m,30m,60m,3h,3h,3h,6h,6h",
    )
    monkeypatch.delenv("MQ_MAX_RETRIES", raising=False)

    get_rabbitmq_settings.cache_clear()
    settings = get_rabbitmq_settings()

    assert settings.retry_delays_seconds == (
        15,
        15,
        30,
        180,
        600,
        1200,
        1800,
        1800,
        1800,
        3600,
        10800,
        10800,
        10800,
        21600,
        21600,
    )
    assert settings.max_retries == 15
    get_rabbitmq_settings.cache_clear()


def test_get_rabbitmq_settings_rejects_invalid_duration_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    测试目的：验证 MQ_RETRY_DELAYS_SECONDS 存在非法时间单位时会抛出配置异常。
    预期结果：读取配置时抛出 ServiceException，阻止错误重试参数生效。
    """
    monkeypatch.setenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
    monkeypatch.setenv("MQ_RETRY_DELAYS_SECONDS", "15s,3x")
    get_rabbitmq_settings.cache_clear()

    with pytest.raises(ServiceException):
        get_rabbitmq_settings()
    get_rabbitmq_settings.cache_clear()
