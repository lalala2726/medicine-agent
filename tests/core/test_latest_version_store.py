import pytest
from redis.exceptions import RedisError

from app.core.exception.exceptions import ServiceException
from app.core.mq.config.settings import RabbitMQSettings
from app.core.mq.state.latest_version_store import (
    build_latest_version_key,
    get_latest_version,
    is_stale_message,
)


class _FakeRedis:
    def __init__(self, values: dict[str, object]) -> None:
        self.values = values

    def get(self, key: str):
        return self.values.get(key)


class _ErrorRedis:
    def get(self, key: str):
        del key
        raise RedisError("mock redis down")


def _build_settings() -> RabbitMQSettings:
    return RabbitMQSettings(
        url="amqp://guest:guest@localhost:5672/",
        exchange="knowledge.import",
        command_queue="knowledge.import.command.q",
        command_routing_key="knowledge.import.command",
        result_routing_key="knowledge.import.result",
        prefetch_count=1,
        latest_version_key_prefix="kb:latest",
    )


def test_build_latest_version_key() -> None:
    """验证 Redis 最新版本键会使用配置前缀。"""
    settings = _build_settings()
    assert build_latest_version_key(biz_key="demo:1", settings=settings) == "kb:latest:demo:1"


def test_get_latest_version_returns_none_when_absent(monkeypatch) -> None:
    """验证 latest key 不存在时返回 None。"""
    monkeypatch.setattr(
        "app.core.mq.state.latest_version_store.get_redis_connection",
        lambda: _FakeRedis(values={}),
    )

    assert get_latest_version(biz_key="demo:1", settings=_build_settings()) is None


def test_get_latest_version_parses_integer_bytes(monkeypatch) -> None:
    """验证 Redis 的 bytes 值可被解析为整数版本号。"""
    monkeypatch.setattr(
        "app.core.mq.state.latest_version_store.get_redis_connection",
        lambda: _FakeRedis(values={"kb:latest:demo:1": b"5"}),
    )

    assert get_latest_version(biz_key="demo:1", settings=_build_settings()) == 5


def test_is_stale_message_true_when_lower_than_latest(monkeypatch) -> None:
    """验证旧版本检查在版本号落后时返回 True。"""
    monkeypatch.setattr(
        "app.core.mq.state.latest_version_store.get_redis_connection",
        lambda: _FakeRedis(values={"kb:latest:demo:1": b"9"}),
    )

    assert is_stale_message(biz_key="demo:1", version=8, settings=_build_settings()) is True
    assert is_stale_message(biz_key="demo:1", version=9, settings=_build_settings()) is False


def test_get_latest_version_raises_service_exception_on_invalid_value(monkeypatch) -> None:
    """验证 Redis 中版本值非法时抛出 ServiceException。"""
    monkeypatch.setattr(
        "app.core.mq.state.latest_version_store.get_redis_connection",
        lambda: _FakeRedis(values={"kb:latest:demo:1": b"abc"}),
    )

    with pytest.raises(ServiceException, match="latest version 不是整数"):
        get_latest_version(biz_key="demo:1", settings=_build_settings())


def test_get_latest_version_raises_service_exception_on_redis_error(monkeypatch) -> None:
    """验证 Redis 调用异常会被包装为 ServiceException。"""
    monkeypatch.setattr(
        "app.core.mq.state.latest_version_store.get_redis_connection",
        lambda: _ErrorRedis(),
    )

    with pytest.raises(ServiceException, match="读取 latest version 失败"):
        get_latest_version(biz_key="demo:1", settings=_build_settings())
