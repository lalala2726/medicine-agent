import asyncio

import pytest

from app.core.exception.exceptions import ServiceException
from app.core.mq.connection import (
    get_rabbitmq_url,
    open_consume_channel,
    open_publish_channel,
)


class _FakeConsumerChannel:
    def __init__(self) -> None:
        self.prefetch_count: int | None = None

    async def set_qos(self, *, prefetch_count: int) -> None:
        self.prefetch_count = prefetch_count


class _FakeConnection:
    def __init__(self, channel) -> None:
        self._channel = channel
        self.publisher_confirms: bool | None = None
        self.entered = 0
        self.exited = 0

    async def __aenter__(self):
        self.entered += 1
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.exited += 1
        return None

    async def channel(self, publisher_confirms: bool = False):
        self.publisher_confirms = publisher_confirms
        return self._channel


class _FakeExchangeType:
    DIRECT = "direct"


class _FakeDeliveryMode:
    PERSISTENT = "persistent"


class _FakeMessage:
    pass


def test_get_rabbitmq_url_requires_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证连接层缺少 RabbitMQ URL 时会抛出异常。"""
    monkeypatch.delenv("RABBITMQ_URL", raising=False)

    with pytest.raises(ServiceException, match="未配置 RABBITMQ_URL"):
        get_rabbitmq_url()


def test_open_publish_channel_opens_confirm_channel(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证发布连接 helper 会打开 confirm channel。"""
    channel = object()
    connection = _FakeConnection(channel)
    captured = {"url": None}

    async def _fake_connect(url: str):
        captured["url"] = url
        return connection

    monkeypatch.setenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
    monkeypatch.setattr(
        "app.core.mq.connection.load_aio_pika_publisher",
        lambda: (_fake_connect, _FakeExchangeType, _FakeMessage, _FakeDeliveryMode),
    )

    async def _run() -> None:
        async with open_publish_channel() as resources:
            assert resources.channel is channel
            assert resources.exchange_type_enum is _FakeExchangeType
            assert resources.message_cls is _FakeMessage
            assert resources.delivery_mode_enum is _FakeDeliveryMode

    asyncio.run(_run())

    assert captured["url"] == "amqp://guest:guest@localhost:5672/"
    assert connection.publisher_confirms is True
    assert connection.entered == 1
    assert connection.exited == 1


def test_open_consume_channel_sets_prefetch(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证消费连接 helper 会设置预取数量。"""
    channel = _FakeConsumerChannel()
    connection = _FakeConnection(channel)
    captured = {"url": None}

    async def _fake_connect(url: str):
        captured["url"] = url
        return connection

    monkeypatch.setenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
    monkeypatch.setattr(
        "app.core.mq.connection.load_aio_pika_consumer",
        lambda: (_fake_connect, _FakeExchangeType),
    )

    async def _run() -> None:
        async with open_consume_channel(prefetch_count=7) as resources:
            assert resources.channel is channel
            assert resources.exchange_type_enum is _FakeExchangeType

    asyncio.run(_run())

    assert captured["url"] == "amqp://guest:guest@localhost:5672/"
    assert channel.prefetch_count == 7
    assert connection.entered == 1
    assert connection.exited == 1
