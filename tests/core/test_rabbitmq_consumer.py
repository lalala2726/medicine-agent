import asyncio
from datetime import datetime, timezone

from app.core.mq.consumer import (
    parse_import_message,
    process_import_message_once,
    process_import_message_with_retry,
)
from app.core.mq.models import KnowledgeImportMessage
from app.rag.chunking import ChunkStrategyType


def _build_message() -> KnowledgeImportMessage:
    """
    测试目的：构造统一的导入消息样例，避免测试用例重复拼装字段。
    预期结果：返回字段完整且可用于消费处理函数的消息对象。
    """
    return KnowledgeImportMessage(
        task_uuid="task-1",
        knowledge_name="demo",
        document_id=7,
        file_url="https://example.com/a.txt",
        chunk_strategy=ChunkStrategyType.CHARACTER,
        chunk_size=500,
        token_size=100,
        created_at=datetime.now(timezone.utc),
    )


def test_parse_import_message_builds_model() -> None:
    """
    测试目的：验证消费者可将原始消息字节正确反序列化为消息模型。
    预期结果：解析后 message 字段与输入 JSON 内容一致。
    """
    body = (
        '{"task_uuid":"task-1","knowledge_name":"demo","document_id":7,'
        '"file_url":"https://example.com/a.txt","chunk_strategy":"character",'
        '"chunk_size":500,"token_size":100,"created_at":"2026-03-03T10:00:00"}'
    ).encode("utf-8")

    message = parse_import_message(body)

    assert message.task_uuid == "task-1"
    assert message.knowledge_name == "demo"
    assert message.document_id == 7
    assert message.file_url == "https://example.com/a.txt"
    assert message.chunk_strategy == ChunkStrategyType.CHARACTER


def test_process_import_message_once_calls_import_service(monkeypatch) -> None:
    """
    测试目的：验证消费者处理单条消息时会调用 import_knowledge_service。
    预期结果：当服务返回 failed_urls 为空时，process_import_message_once 返回 True。
    """
    called: dict = {}

    def _fake_import_service(
            knowledge_name,
            document_id,
            file_url,
            chunk_strategy,
            chunk_size,
            token_size,
    ):
        called["knowledge_name"] = knowledge_name
        called["document_id"] = document_id
        called["file_url"] = file_url
        called["chunk_strategy"] = chunk_strategy
        called["chunk_size"] = chunk_size
        called["token_size"] = token_size
        return {"results": [{"ok": True}], "failed_urls": []}

    monkeypatch.setattr(
        "app.core.mq.consumer.import_knowledge_service",
        _fake_import_service,
    )
    message = _build_message()
    success = process_import_message_once(message)

    assert success is True
    assert called["knowledge_name"] == "demo"
    assert called["document_id"] == 7
    assert called["file_url"] == ["https://example.com/a.txt"]


def test_process_import_message_with_retry_retries_until_success() -> None:
    """
    测试目的：验证消费者失败后会按重试策略再次执行，直到成功。
    预期结果：处理函数在第 3 次返回成功，总调用次数为 3 次。
    """
    message = _build_message()
    called = {"count": 0}
    slept: list[float] = []

    def _fake_process(_message: KnowledgeImportMessage) -> bool:
        called["count"] += 1
        return called["count"] >= 3

    async def _fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    success = asyncio.run(
        process_import_message_with_retry(
            message,
            max_retries=3,
            retry_delays_seconds=(5, 30, 120),
            process_func=_fake_process,
            sleep_func=_fake_sleep,
        )
    )

    assert success is True
    assert called["count"] == 3
    assert slept == [5.0, 30.0]


def test_process_import_message_with_retry_stops_after_max_retries() -> None:
    """
    测试目的：验证消费者在达到最大重试次数后停止重试并返回失败。
    预期结果：处理函数总调用次数为 max_retries+1，函数返回 False。
    """
    message = _build_message()
    called = {"count": 0}
    slept: list[float] = []

    def _always_fail(_message: KnowledgeImportMessage) -> bool:
        called["count"] += 1
        return False

    async def _fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    success = asyncio.run(
        process_import_message_with_retry(
            message,
            max_retries=2,
            retry_delays_seconds=(5, 30, 120),
            process_func=_always_fail,
            sleep_func=_fake_sleep,
        )
    )

    assert success is False
    assert called["count"] == 3
    assert slept == [5.0, 30.0]
