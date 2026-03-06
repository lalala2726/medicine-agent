import asyncio
from datetime import datetime, timezone

from app.core.mq.config.import_settings import ImportRabbitMQSettings
from app.core.mq.consumers.import_consumer import _handle_incoming_message, parse_import_command
from app.core.mq.contracts.import_models import (
    ImportResultStage,
    KnowledgeImportCommandMessage,
    ProcessingStageDetail,
)
from app.rag.chunking import ChunkStrategyType
from app.schemas.knowledge_import import (
    ImportSingleFileFailedResult,
    ImportSingleFileSuccessResult,
)


def _build_command() -> KnowledgeImportCommandMessage:
    return KnowledgeImportCommandMessage(
        task_uuid="task-1",
        biz_key="demo:7",
        version=3,
        knowledge_name="demo",
        document_id=7,
        file_url="https://example.com/a.txt",
        embedding_model="text-embedding-v4",
        chunk_strategy=ChunkStrategyType.CHARACTER,
        chunk_size=500,
        token_size=100,
        created_at=datetime.now(timezone.utc),
    )


def _build_settings() -> ImportRabbitMQSettings:
    return ImportRabbitMQSettings(
        url="amqp://guest:guest@localhost:5672/",
        exchange="knowledge.import",
        command_queue="knowledge.import.command.q",
        command_routing_key="knowledge.import.command",
        result_routing_key="knowledge.import.result",
        prefetch_count=1,
        latest_version_key_prefix="kb:latest",
    )


class _FakeIncoming:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.ack_called = 0

    async def ack(self) -> None:
        self.ack_called += 1


def test_parse_import_command_builds_model() -> None:
    """验证消费者可正确解析合法命令消息。"""
    body = (
        '{"message_type":"knowledge_import_command","task_uuid":"task-1","biz_key":"demo:7",'
        '"version":3,"knowledge_name":"demo","document_id":7,'
        '"file_url":"https://example.com/a.txt","embedding_model":"text-embedding-v4",'
        '"chunk_strategy":"character","chunk_size":500,"token_size":100,'
        '"created_at":"2026-03-05T10:00:00"}'
    ).encode("utf-8")

    message = parse_import_command(body)

    assert message.task_uuid == "task-1"
    assert message.biz_key == "demo:7"
    assert message.version == 3
    assert message.chunk_strategy == ChunkStrategyType.CHARACTER


def test_parse_import_command_allows_null_token_size_for_character() -> None:
    """验证 character 切片下 token_size 为 null 时会回退默认值。"""
    body = (
        '{"message_type":"knowledge_import_command","task_uuid":"task-1","biz_key":"demo:7",'
        '"version":3,"knowledge_name":"demo","document_id":7,'
        '"file_url":"https://example.com/a.txt","embedding_model":"text-embedding-v4",'
        '"chunk_strategy":"character","chunk_size":500,"token_size":null,'
        '"created_at":"2026-03-05T10:00:00"}'
    ).encode("utf-8")

    message = parse_import_command(body)

    assert message.chunk_strategy == ChunkStrategyType.CHARACTER
    assert message.chunk_size == 500
    assert message.token_size == 100


def test_handle_incoming_message_drops_stale_command(monkeypatch) -> None:
    """验证旧版本命令会被 ACK 丢弃且不发送结果事件。"""
    command = _build_command()
    incoming = _FakeIncoming(command.model_dump_json().encode("utf-8"))
    published = {"count": 0}

    monkeypatch.setattr("app.core.mq.consumers.import_consumer.is_stale", lambda **_kwargs: True)

    async def _fake_publish(_event) -> bool:
        published["count"] += 1
        return True

    monkeypatch.setattr("app.core.mq.consumers.import_consumer.publish_import_result", _fake_publish)

    asyncio.run(_handle_incoming_message(incoming, _build_settings()))

    assert incoming.ack_called == 1
    assert published["count"] == 0


def test_handle_incoming_message_emits_stage_events_on_success(monkeypatch) -> None:
    """验证成功链路会依次发送 STARTED、PROCESSING 明细和 COMPLETED。"""
    command = _build_command()
    incoming = _FakeIncoming(command.model_dump_json().encode("utf-8"))
    events = []

    monkeypatch.setattr("app.core.mq.consumers.import_consumer.is_stale", lambda **_kwargs: False)

    async def _fake_publish(event) -> bool:
        events.append(event)
        return True

    def _fake_import_single_file(**kwargs):
        on_processing_stage = kwargs["on_processing_stage"]
        on_processing_stage(ProcessingStageDetail.DOWNLOADING)
        on_processing_stage(ProcessingStageDetail.PARSING)
        on_processing_stage(ProcessingStageDetail.CHUNKING)
        on_processing_stage(ProcessingStageDetail.EMBEDDING)
        on_processing_stage(ProcessingStageDetail.INSERTING)
        return ImportSingleFileSuccessResult(
            file_url=kwargs["url"],
            filename="a.txt",
            source_extension=".txt",
            file_kind="text",
            mime_type="text/plain",
            text_length=100,
            chunk_count=3,
            vector_count=3,
            insert_batches=1,
            embedding_model=kwargs["embedding_model"],
            embedding_dim=1024,
            chunks=[],
        )

    monkeypatch.setattr("app.core.mq.consumers.import_consumer.publish_import_result", _fake_publish)
    monkeypatch.setattr("app.core.mq.consumers.import_consumer.import_single_file", _fake_import_single_file)

    asyncio.run(_handle_incoming_message(incoming, _build_settings()))

    assert incoming.ack_called == 1
    assert [event.stage for event in events] == [
        ImportResultStage.STARTED,
        ImportResultStage.PROCESSING,
        ImportResultStage.PROCESSING,
        ImportResultStage.PROCESSING,
        ImportResultStage.PROCESSING,
        ImportResultStage.PROCESSING,
        ImportResultStage.COMPLETED,
    ]
    assert [event.stage_detail for event in events[1:6]] == [
        ProcessingStageDetail.DOWNLOADING,
        ProcessingStageDetail.PARSING,
        ProcessingStageDetail.CHUNKING,
        ProcessingStageDetail.EMBEDDING,
        ProcessingStageDetail.INSERTING,
    ]
    assert events[-1].chunk_count == 3
    assert events[-1].vector_count == 3
    assert all(event.task_uuid == "task-1" for event in events)
    assert all(event.version == 3 for event in events)


def test_handle_incoming_message_emits_failed_without_retry(monkeypatch) -> None:
    """验证处理失败时会立即发送 FAILED 事件且不重试。"""
    command = _build_command()
    incoming = _FakeIncoming(command.model_dump_json().encode("utf-8"))
    events = []

    monkeypatch.setattr("app.core.mq.consumers.import_consumer.is_stale", lambda **_kwargs: False)

    async def _fake_publish(event) -> bool:
        events.append(event)
        return True

    def _fake_import_single_file(**kwargs):
        return ImportSingleFileFailedResult(
            file_url=kwargs["url"],
            filename="a.txt",
            error="mock failure",
            embedding_model=kwargs["embedding_model"],
            embedding_dim=1024,
        )

    monkeypatch.setattr("app.core.mq.consumers.import_consumer.publish_import_result", _fake_publish)
    monkeypatch.setattr("app.core.mq.consumers.import_consumer.import_single_file", _fake_import_single_file)

    asyncio.run(_handle_incoming_message(incoming, _build_settings()))

    assert incoming.ack_called == 1
    assert [event.stage for event in events] == [
        ImportResultStage.STARTED,
        ImportResultStage.FAILED,
    ]
    assert events[-1].message == "mock failure"


def test_handle_incoming_message_acks_when_result_publish_fails(monkeypatch) -> None:
    """验证结果发布失败时消息仍会 ACK。"""
    command = _build_command()
    incoming = _FakeIncoming(command.model_dump_json().encode("utf-8"))

    monkeypatch.setattr("app.core.mq.consumers.import_consumer.is_stale", lambda **_kwargs: False)

    async def _always_fail_publish(_event) -> bool:
        return False

    def _fake_import_single_file(**kwargs):
        return ImportSingleFileSuccessResult(
            file_url=kwargs["url"],
            filename="a.txt",
            source_extension=".txt",
            file_kind="text",
            mime_type="text/plain",
            text_length=10,
            chunk_count=1,
            vector_count=1,
            insert_batches=1,
            embedding_model=kwargs["embedding_model"],
            embedding_dim=1024,
            chunks=[],
        )

    monkeypatch.setattr("app.core.mq.consumers.import_consumer.publish_import_result", _always_fail_publish)
    monkeypatch.setattr("app.core.mq.consumers.import_consumer.import_single_file", _fake_import_single_file)

    asyncio.run(_handle_incoming_message(incoming, _build_settings()))

    assert incoming.ack_called == 1


def test_handle_incoming_message_acks_invalid_payload() -> None:
    """验证非法消息体会被 ACK 并丢弃。"""
    incoming = _FakeIncoming(b"{invalid json")
    asyncio.run(_handle_incoming_message(incoming, _build_settings()))
    assert incoming.ack_called == 1
