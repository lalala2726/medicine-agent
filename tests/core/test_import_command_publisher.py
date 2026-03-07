import asyncio
from datetime import datetime, timezone

from app.core.mq.config.document.import_settings import ImportRabbitMQSettings
from app.core.mq.contracts.document.import_models import KnowledgeImportCommandMessage
from app.core.mq.producers.document.import_command_publisher import (
    publish_import_commands,
)
from app.rag.chunking import ChunkStrategyType


class _FakeExchange:
    def __init__(self) -> None:
        self.published: list[tuple[object, str]] = []

    async def publish(self, message, routing_key: str) -> None:
        self.published.append((message, routing_key))


class _FakeQueue:
    def __init__(self) -> None:
        self.bind_calls: list[tuple[object, str]] = []

    async def bind(self, exchange, routing_key: str) -> None:
        self.bind_calls.append((exchange, routing_key))


class _FakeChannel:
    def __init__(self, exchange: _FakeExchange, queue: _FakeQueue) -> None:
        self.exchange = exchange
        self.queue = queue

    async def declare_exchange(self, *_args, **_kwargs):
        return self.exchange

    async def declare_queue(self, *_args, **_kwargs):
        return self.queue


class _FakeConnection:
    def __init__(self, channel: _FakeChannel) -> None:
        self._channel = channel
        self.publisher_confirms: bool | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def channel(self, publisher_confirms: bool = True):
        self.publisher_confirms = publisher_confirms
        return self._channel


class _FakeExchangeType:
    DIRECT = "direct"


class _FakeDeliveryMode:
    PERSISTENT = "persistent"


class _FakeMessage:
    def __init__(self, body: bytes, content_type: str, delivery_mode: str) -> None:
        self.body = body
        self.content_type = content_type
        self.delivery_mode = delivery_mode


def test_publish_import_commands_serializes_and_publishes(monkeypatch) -> None:
    """验证命令发布器会按命令路由键投递消息。"""
    exchange = _FakeExchange()
    queue = _FakeQueue()
    channel = _FakeChannel(exchange, queue)
    connection = _FakeConnection(channel)
    settings = ImportRabbitMQSettings(
        url="amqp://guest:guest@localhost:5672/",
        exchange="knowledge.import",
        command_queue="knowledge.import.command.q",
        command_routing_key="knowledge.import.command",
        result_routing_key="knowledge.import.result",
        prefetch_count=1,
        latest_version_key_prefix="kb:latest",
    )

    async def _fake_connect(_url: str):
        return connection

    monkeypatch.setattr(
        "app.core.mq.producers.document.import_command_publisher.get_import_settings",
        lambda: settings,
    )
    monkeypatch.setattr(
        "app.core.mq.producers.document.import_command_publisher.load_aio_pika_publisher",
        lambda: (_fake_connect, _FakeExchangeType, _FakeMessage, _FakeDeliveryMode),
    )

    message = KnowledgeImportCommandMessage(
        task_uuid="task-1",
        biz_key="demo:1",
        version=1,
        knowledge_name="demo",
        document_id=1,
        file_url="https://example.com/a.txt",
        embedding_model="text-embedding-v4",
        chunk_strategy=ChunkStrategyType.CHARACTER,
        chunk_size=500,
        token_size=100,
        created_at=datetime.now(timezone.utc),
    )
    asyncio.run(publish_import_commands([message]))

    assert len(exchange.published) == 1
    _, routing_key = exchange.published[0]
    assert routing_key == "knowledge.import.command"
    assert queue.bind_calls[0][1] == "knowledge.import.command"
    assert connection.publisher_confirms is True


def test_publish_import_commands_noop_when_empty(monkeypatch) -> None:
    """验证输入为空时不会触发 MQ 连接。"""
    called = {"load": False}

    monkeypatch.setattr(
        "app.core.mq.producers.document.import_command_publisher.load_aio_pika_publisher",
        lambda: (_mark_load_called(called), _FakeExchangeType, _FakeMessage, _FakeDeliveryMode),
    )
    asyncio.run(publish_import_commands([]))
    assert called["load"] is False


def _mark_load_called(called: dict[str, bool]):
    """构造用于空输入断言的假连接函数。"""

    async def _fake_connect(_url: str):
        called["load"] = True
        raise RuntimeError("should not connect")

    return _fake_connect
