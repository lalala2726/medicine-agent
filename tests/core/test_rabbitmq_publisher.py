import asyncio
from datetime import datetime, timezone

from app.core.mq.models import KnowledgeImportMessage
from app.core.mq.publisher import publish_import_messages
from app.core.mq.settings import RabbitMQSettings
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
        self.exchange_declared: tuple[str, str, bool] | None = None
        self.queue_declared: tuple[str, bool] | None = None

    async def declare_exchange(self, name: str, exchange_type, durable: bool):
        self.exchange_declared = (name, str(exchange_type), durable)
        return self.exchange

    async def declare_queue(self, name: str, durable: bool):
        self.queue_declared = (name, durable)
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


def test_publish_import_messages_serializes_and_publishes(monkeypatch) -> None:
    """
    测试目的：验证发布器会将消息序列化并按配置路由发布到 RabbitMQ。
    预期结果：发布次数与输入消息数一致，routing_key 使用配置值。
    """
    exchange = _FakeExchange()
    queue = _FakeQueue()
    channel = _FakeChannel(exchange, queue)
    connection = _FakeConnection(channel)
    settings = RabbitMQSettings(
        url="amqp://guest:guest@localhost:5672/",
        exchange="knowledge.import",
        queue="knowledge.import.submit.q",
        routing_key="knowledge.import.submit",
        prefetch_count=1,
        max_retries=3,
        retry_delays_seconds=(5, 30, 120),
    )

    async def _fake_connect(_url: str):
        return connection

    monkeypatch.setattr(
        "app.core.mq.publisher.get_rabbitmq_settings",
        lambda: settings,
    )
    monkeypatch.setattr(
        "app.core.mq.publisher._load_aio_pika",
        lambda: (_fake_connect, _FakeExchangeType, _FakeMessage, _FakeDeliveryMode),
    )

    message = KnowledgeImportMessage(
        task_uuid="task-1",
        knowledge_name="demo",
        document_id=1,
        file_url="https://example.com/a.txt",
        chunk_strategy=ChunkStrategyType.CHARACTER,
        chunk_size=500,
        token_size=100,
        created_at=datetime.now(timezone.utc),
    )
    asyncio.run(publish_import_messages([message]))

    assert len(exchange.published) == 1
    _, routing_key = exchange.published[0]
    assert routing_key == "knowledge.import.submit"
    assert queue.bind_calls[0][1] == "knowledge.import.submit"
    assert connection.publisher_confirms is True


def test_publish_import_messages_noop_when_empty(monkeypatch) -> None:
    """
    测试目的：验证空消息列表不会触发 RabbitMQ 连接和发布动作。
    预期结果：connect_robust 不被调用，函数正常返回。
    """
    called = {"load": False}

    monkeypatch.setattr(
        "app.core.mq.publisher._load_aio_pika",
        lambda: (_mark_load_called(called), _FakeExchangeType, _FakeMessage, _FakeDeliveryMode),
    )
    asyncio.run(publish_import_messages([]))
    assert called["load"] is False


def _mark_load_called(called: dict[str, bool]):
    """
    测试目的：构造用于断言懒加载是否触发的 connect_robust 假函数。
    预期结果：当函数被调用时，called["load"] 被置为 True。
    """

    async def _fake_connect(_url: str):
        called["load"] = True
        raise RuntimeError("should not connect")

    return _fake_connect
