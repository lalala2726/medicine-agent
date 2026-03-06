import asyncio
from datetime import datetime, timezone

from app.core.mq.config.chunk_rebuild_settings import ChunkRebuildRabbitMQSettings
from app.core.mq.consumers.chunk_rebuild_consumer import (
    _handle_incoming_message,
    parse_chunk_rebuild_command,
)
from app.core.mq.contracts.chunk_rebuild_models import (
    ChunkRebuildResultStage,
    KnowledgeChunkRebuildCommandMessage,
)
from app.services.chunk_rebuild_service import (
    ChunkRebuildMessageStaleError,
    ChunkRebuildSuccessResult,
)


def _build_command() -> KnowledgeChunkRebuildCommandMessage:
    return KnowledgeChunkRebuildCommandMessage(
        task_uuid="task-1",
        knowledge_name="demo_kb",
        document_id=7,
        vector_id=101,
        version=3,
        content="updated chunk",
        embedding_model="text-embedding-v4",
        created_at=datetime.now(timezone.utc),
    )


def _build_settings() -> ChunkRebuildRabbitMQSettings:
    return ChunkRebuildRabbitMQSettings(
        url="amqp://guest:guest@localhost:5672/",
        exchange="knowledge.chunk_rebuild",
        command_queue="knowledge.chunk_rebuild.command.q",
        command_routing_key="knowledge.chunk_rebuild.command",
        result_routing_key="knowledge.chunk_rebuild.result",
        prefetch_count=1,
        latest_version_key_prefix="kb:chunk_edit:latest_version",
    )


class _FakeIncoming:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.ack_called = 0

    async def ack(self) -> None:
        self.ack_called += 1


def test_parse_chunk_rebuild_command_builds_model() -> None:
    """验证消费者可正确解析合法切片重建命令消息。"""
    body = (
        '{"message_type":"knowledge_chunk_rebuild_command","task_uuid":"task-1",'
        '"knowledge_name":"demo_kb","document_id":7,"vector_id":101,"version":3,'
        '"content":"updated chunk","embedding_model":"text-embedding-v4",'
        '"created_at":"2026-03-05T10:00:00"}'
    ).encode("utf-8")

    message = parse_chunk_rebuild_command(body)

    assert message.task_uuid == "task-1"
    assert message.vector_id == 101
    assert message.version == 3
    assert message.content == "updated chunk"


def test_handle_incoming_message_emits_started_and_completed(monkeypatch) -> None:
    """验证成功链路会依次发送 STARTED、COMPLETED 并 ACK。"""
    command = _build_command()
    incoming = _FakeIncoming(command.model_dump_json().encode("utf-8"))
    events = []

    async def _fake_publish(event) -> bool:
        events.append(event)
        return True

    def _fake_rebuild_document_chunk(**_kwargs):
        return ChunkRebuildSuccessResult(embedding_dim=1024)

    monkeypatch.setattr(
        "app.core.mq.consumers.chunk_rebuild_consumer.publish_chunk_rebuild_result",
        _fake_publish,
    )
    monkeypatch.setattr(
        "app.core.mq.consumers.chunk_rebuild_consumer.get_latest_version",
        lambda **_kwargs: 3,
    )
    monkeypatch.setattr(
        "app.core.mq.consumers.chunk_rebuild_consumer.rebuild_document_chunk",
        _fake_rebuild_document_chunk,
    )

    asyncio.run(_handle_incoming_message(incoming, _build_settings()))

    assert incoming.ack_called == 1
    assert [event.stage for event in events] == [
        ChunkRebuildResultStage.STARTED,
        ChunkRebuildResultStage.COMPLETED,
    ]
    assert all(event.version == 3 for event in events)
    assert events[-1].embedding_dim == 1024


def test_handle_incoming_message_emits_started_and_failed(monkeypatch) -> None:
    """验证处理失败时会发送 FAILED 事件并 ACK。"""
    command = _build_command()
    incoming = _FakeIncoming(command.model_dump_json().encode("utf-8"))
    events = []

    async def _fake_publish(event) -> bool:
        events.append(event)
        return True

    def _fake_rebuild_document_chunk(**_kwargs):
        raise RuntimeError("mock failure")

    monkeypatch.setattr(
        "app.core.mq.consumers.chunk_rebuild_consumer.publish_chunk_rebuild_result",
        _fake_publish,
    )
    monkeypatch.setattr(
        "app.core.mq.consumers.chunk_rebuild_consumer.get_latest_version",
        lambda **_kwargs: 3,
    )
    monkeypatch.setattr(
        "app.core.mq.consumers.chunk_rebuild_consumer.rebuild_document_chunk",
        _fake_rebuild_document_chunk,
    )

    asyncio.run(_handle_incoming_message(incoming, _build_settings()))

    assert incoming.ack_called == 1
    assert [event.stage for event in events] == [
        ChunkRebuildResultStage.STARTED,
        ChunkRebuildResultStage.FAILED,
    ]
    assert events[-1].message == "mock failure"


def test_handle_incoming_message_does_not_ack_when_result_publish_fails(monkeypatch) -> None:
    """验证最终结果发布失败时原命令不会 ACK。"""
    command = _build_command()
    incoming = _FakeIncoming(command.model_dump_json().encode("utf-8"))
    published = {"count": 0}

    async def _fake_publish(_event) -> bool:
        published["count"] += 1
        return published["count"] == 1

    def _fake_rebuild_document_chunk(**_kwargs):
        return ChunkRebuildSuccessResult(embedding_dim=1024)

    monkeypatch.setattr(
        "app.core.mq.consumers.chunk_rebuild_consumer.publish_chunk_rebuild_result",
        _fake_publish,
    )
    monkeypatch.setattr(
        "app.core.mq.consumers.chunk_rebuild_consumer.get_latest_version",
        lambda **_kwargs: 3,
    )
    monkeypatch.setattr(
        "app.core.mq.consumers.chunk_rebuild_consumer.rebuild_document_chunk",
        _fake_rebuild_document_chunk,
    )

    asyncio.run(_handle_incoming_message(incoming, _build_settings()))

    assert incoming.ack_called == 0
    assert published["count"] == 2


def test_handle_incoming_message_acks_invalid_payload() -> None:
    """验证非法消息体会被 ACK 并丢弃。"""
    incoming = _FakeIncoming(b"{invalid json")
    asyncio.run(_handle_incoming_message(incoming, _build_settings()))
    assert incoming.ack_called == 1


def test_handle_incoming_message_drops_stale_command_and_emits_failed(
        monkeypatch,
) -> None:
    """验证旧版本命令会被丢弃并发送 FAILED 结果。"""
    command = _build_command()
    incoming = _FakeIncoming(command.model_dump_json().encode("utf-8"))
    events = []

    async def _fake_publish(event) -> bool:
        events.append(event)
        return True

    monkeypatch.setattr(
        "app.core.mq.consumers.chunk_rebuild_consumer.publish_chunk_rebuild_result",
        _fake_publish,
    )
    monkeypatch.setattr(
        "app.core.mq.consumers.chunk_rebuild_consumer.get_latest_version",
        lambda **_kwargs: 5,
    )

    asyncio.run(_handle_incoming_message(incoming, _build_settings()))

    assert incoming.ack_called == 1
    assert [event.stage for event in events] == [ChunkRebuildResultStage.FAILED]
    assert "latest_version=5" in events[0].message


def test_handle_incoming_message_emits_failed_when_write_phase_becomes_stale(
        monkeypatch,
) -> None:
    """验证写入前发现任务已过期时会发送 FAILED 并 ACK。"""
    command = _build_command()
    incoming = _FakeIncoming(command.model_dump_json().encode("utf-8"))
    events = []

    async def _fake_publish(event) -> bool:
        events.append(event)
        return True

    monkeypatch.setattr(
        "app.core.mq.consumers.chunk_rebuild_consumer.publish_chunk_rebuild_result",
        _fake_publish,
    )
    monkeypatch.setattr(
        "app.core.mq.consumers.chunk_rebuild_consumer.get_latest_version",
        lambda **_kwargs: 3,
    )
    monkeypatch.setattr(
        "app.core.mq.consumers.chunk_rebuild_consumer.rebuild_document_chunk",
        lambda **_kwargs: (_ for _ in ()).throw(
            ChunkRebuildMessageStaleError(vector_id=101, version=3, latest_version=4)
        ),
    )

    asyncio.run(_handle_incoming_message(incoming, _build_settings()))

    assert incoming.ack_called == 1
    assert [event.stage for event in events] == [
        ChunkRebuildResultStage.STARTED,
        ChunkRebuildResultStage.FAILED,
    ]
    assert "latest_version=4" in events[-1].message
