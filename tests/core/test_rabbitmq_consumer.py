import asyncio
from datetime import datetime, timezone

from app.core.mq.consumer import (
    CallbackAttemptResult,
    CallbackTracker,
    _handle_incoming_message,
    _is_callback_success,
    parse_import_message,
    process_import_message_once,
    process_import_message_with_retry,
    send_callback_with_retry,
)
from app.core.mq.models import (
    CallbackStage,
    KnowledgeImportCallbackPayload,
    KnowledgeImportMessage,
)
from app.core.mq.settings import RabbitMQSettings
from app.rag.chunking import ChunkStrategyType
from app.schemas.knowledge_import import ImportSingleFileSuccessResult


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
        embedding_model="text-embedding-v4",
        chunk_strategy=ChunkStrategyType.CHARACTER,
        chunk_size=500,
        token_size=100,
        created_at=datetime.now(timezone.utc),
    )


def _build_settings() -> RabbitMQSettings:
    """
    测试目的：构造消费者测试所需的最小 MQ 配置对象。
    预期结果：返回可用于回调重试函数的配置实例。
    """
    return RabbitMQSettings(
        url="amqp://guest:guest@localhost:5672/",
        exchange="knowledge.import",
        queue="knowledge.import.submit.q",
        routing_key="knowledge.import.submit",
        prefetch_count=1,
        max_retries=2,
        retry_delays_seconds=(5, 30),
        callback_url="http://localhost:8083/api/knowledge/callback",
        callback_timeout_seconds=5,
        callback_max_retries=2,
        callback_retry_delays_seconds=(5, 30),
    )


def _build_callback_payload(
        status: str = "COMPLETED",
) -> KnowledgeImportCallbackPayload:
    """
    测试目的：构造回调参数样例，供回调重试测试复用。
    预期结果：返回字段完整的 payload 对象。
    """
    return KnowledgeImportCallbackPayload.build(
        task_uuid="task-1",
        knowledge_name="demo",
        document_id=7,
        file_url="https://example.com/a.txt",
        status=status,
        message="ok" if status == CallbackStage.COMPLETED.value else "failed",
        embedding_model="text-embedding-v4",
        embedding_dim=1024,
        chunk_strategy="character",
        chunk_size=500,
        token_size=100,
        chunk_count=3,
        vector_count=3,
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# parse_import_message
# ---------------------------------------------------------------------------


def test_parse_import_message_builds_model() -> None:
    """
    测试目的：验证消费者可将原始消息字节正确反序列化为消息模型。
    预期结果：解析后 message 字段与输入 JSON 内容一致。
    """
    body = (
        '{"task_uuid":"task-1","knowledge_name":"demo","document_id":7,'
        '"file_url":"https://example.com/a.txt","embedding_model":"text-embedding-v4",'
        '"chunk_strategy":"character","chunk_size":500,"token_size":100,'
        '"created_at":"2026-03-03T10:00:00"}'
    ).encode("utf-8")

    message = parse_import_message(body)

    assert message.task_uuid == "task-1"
    assert message.knowledge_name == "demo"
    assert message.document_id == 7
    assert message.file_url == "https://example.com/a.txt"
    assert message.embedding_model == "text-embedding-v4"
    assert message.chunk_strategy == ChunkStrategyType.CHARACTER


# ---------------------------------------------------------------------------
# process_import_message_once
# ---------------------------------------------------------------------------


def test_process_import_message_once_returns_success_payload(monkeypatch) -> None:
    """
    测试目的：验证单条导入处理成功时会返回 COMPLETED 回调参数。
    预期结果：payload 状态为 COMPLETED，且包含 chunk_count/vector_count 信息。
    """
    called: dict = {}

    def _fake_import_single_file(
            url,
            knowledge_name,
            document_id,
            embedding_model,
            chunk_strategy,
            chunk_size,
            token_size,
            task_uuid="-",
    ):
        called["url"] = url
        called["knowledge_name"] = knowledge_name
        called["document_id"] = document_id
        called["embedding_model"] = embedding_model
        return ImportSingleFileSuccessResult(
            file_url=url,
            filename="a.txt",
            source_extension=".txt",
            file_kind="text",
            mime_type="text/plain",
            text_length=100,
            chunk_count=3,
            vector_count=3,
            insert_batches=1,
            embedding_model=embedding_model,
            embedding_dim=1024,
            chunks=[],
        )

    monkeypatch.setattr(
        "app.core.mq.consumer.import_single_file",
        _fake_import_single_file,
    )
    message = _build_message()
    payload = process_import_message_once(message)

    assert payload.status == CallbackStage.COMPLETED.value
    assert payload.chunk_count == 3
    assert payload.vector_count == 3
    assert called["knowledge_name"] == "demo"
    assert called["document_id"] == 7
    assert called["url"] == "https://example.com/a.txt"
    assert called["embedding_model"] == "text-embedding-v4"


# ---------------------------------------------------------------------------
# process_import_message_with_retry
# ---------------------------------------------------------------------------


def test_process_import_message_with_retry_retries_until_success() -> None:
    """
    测试目的：验证消费者失败后会按重试策略再次执行，直到成功。
    预期结果：处理函数在第 3 次返回 COMPLETED，总调用次数为 3 次。
    """
    message = _build_message()
    called = {"count": 0}
    slept: list[float] = []

    def _fake_process(
            _message: KnowledgeImportMessage, **_kwargs
    ) -> KnowledgeImportCallbackPayload:
        called["count"] += 1
        if called["count"] < 3:
            return _build_callback_payload("FAILED")
        return _build_callback_payload("COMPLETED")

    async def _fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    payload = asyncio.run(
        process_import_message_with_retry(
            message,
            max_retries=3,
            retry_delays_seconds=(5, 30, 120),
            process_func=_fake_process,
            sleep_func=_fake_sleep,
        )
    )

    assert payload.status == CallbackStage.COMPLETED.value
    assert called["count"] == 3
    assert slept == [5.0, 30.0]


def test_process_import_message_with_retry_stops_after_max_retries() -> None:
    """
    测试目的：验证消费者在达到最大重试次数后停止重试并返回失败 payload。
    预期结果：处理函数总调用次数为 max_retries+1，最终状态为 FAILED。
    """
    message = _build_message()
    called = {"count": 0}
    slept: list[float] = []

    def _always_fail(
            _message: KnowledgeImportMessage, **_kwargs
    ) -> KnowledgeImportCallbackPayload:
        called["count"] += 1
        return _build_callback_payload("FAILED")

    async def _fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    payload = asyncio.run(
        process_import_message_with_retry(
            message,
            max_retries=2,
            retry_delays_seconds=(5, 30, 120),
            process_func=_always_fail,
            sleep_func=_fake_sleep,
        )
    )

    assert payload.status == CallbackStage.FAILED.value
    assert called["count"] == 3
    assert slept == [5.0, 30.0]


# ---------------------------------------------------------------------------
# _is_callback_success
# ---------------------------------------------------------------------------


def test_is_callback_success_requires_http_200_and_exact_success_body() -> None:
    """
    测试目的：验证回调成功判定必须同时满足 200 状态码和响应体严格等于 SUCCESS。
    预期结果：仅 (200, "SUCCESS") 判定为 True，其余场景均为 False。
    """
    assert _is_callback_success(200, "SUCCESS") is True
    assert _is_callback_success(200, " SUCCESS \n") is True
    assert _is_callback_success(200, "OK") is False
    assert _is_callback_success(500, "SUCCESS") is False


# ---------------------------------------------------------------------------
# send_callback_with_retry
# ---------------------------------------------------------------------------


def test_send_callback_with_retry_stops_after_window() -> None:
    """
    测试目的：验证回调失败会按重试窗口执行，超限后返回 False。
    预期结果：回调函数调用 max_retries+1 次，sleep 次数为 max_retries。
    """
    payload = _build_callback_payload("FAILED")
    settings = _build_settings()
    called = {"count": 0}
    slept: list[float] = []

    async def _always_fail(
            _payload: KnowledgeImportCallbackPayload,
            _settings: RabbitMQSettings,
    ) -> CallbackAttemptResult:
        called["count"] += 1
        return CallbackAttemptResult(
            success=False,
            status_code=404,
            body_snippet="NOT_FOUND",
            error_message=None,
        )

    async def _fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    callback_ok = asyncio.run(
        send_callback_with_retry(
            payload,
            settings=settings,
            max_retries=2,
            retry_delays_seconds=(5, 30, 120),
            send_func=_always_fail,
            sleep_func=_fake_sleep,
        )
    )

    assert callback_ok is False
    assert called["count"] == 3
    assert slept == [5.0, 30.0]


def test_send_callback_with_retry_returns_when_first_attempt_success() -> None:
    """
    测试目的：验证回调第一次命中成功条件时会立即结束，不会进入后续等待。
    预期结果：回调函数仅调用一次，sleep 不会被调用，函数返回 True。
    """
    payload = _build_callback_payload("COMPLETED")
    settings = _build_settings()
    called = {"count": 0}
    slept: list[float] = []

    async def _success_once(
            _payload: KnowledgeImportCallbackPayload,
            _settings: RabbitMQSettings,
    ) -> CallbackAttemptResult:
        called["count"] += 1
        return CallbackAttemptResult(
            success=True,
            status_code=200,
            body_snippet="SUCCESS",
            error_message=None,
        )

    async def _fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    callback_ok = asyncio.run(
        send_callback_with_retry(
            payload,
            settings=settings,
            max_retries=3,
            retry_delays_seconds=(5, 30, 120),
            send_func=_success_once,
            sleep_func=_fake_sleep,
        )
    )

    assert callback_ok is True
    assert called["count"] == 1
    assert slept == []


# ---------------------------------------------------------------------------
# CallbackTracker
# ---------------------------------------------------------------------------


def test_callback_tracker_merges_state_only_sends_latest() -> None:
    """
    测试目的：验证 CallbackTracker 状态合并语义 — STARTED 失败后推进到 PROCESSING，
    flush 只发 PROCESSING 不补发 STARTED。
    预期结果：发送函数被调用两次（STARTED 失败 + PROCESSING 成功），最终发送的是 PROCESSING。
    """
    settings = _build_settings()
    sent_payloads: list[KnowledgeImportCallbackPayload] = []

    call_count = {"n": 0}

    async def _fake_send(
            payload: KnowledgeImportCallbackPayload,
            _settings: RabbitMQSettings,
    ) -> CallbackAttemptResult:
        call_count["n"] += 1
        sent_payloads.append(payload)
        # 第一次调用（STARTED）失败
        if call_count["n"] == 1:
            return CallbackAttemptResult(
                success=False, status_code=502, body_snippet="BAD_GW", error_message=None
            )
        # 第二次调用（PROCESSING）成功
        return CallbackAttemptResult(
            success=True, status_code=200, body_snippet="SUCCESS", error_message=None
        )

    tracker = CallbackTracker(task_uuid="test-merge")

    # advance STARTED → flush (失败)
    started_payload = _build_callback_payload("STARTED")
    tracker.advance(started_payload)
    r1 = asyncio.run(tracker.flush(settings, send_func=_fake_send))
    assert r1 is False

    # advance PROCESSING → flush (成功)
    processing_payload = _build_callback_payload("PROCESSING")
    tracker.advance(processing_payload)
    r2 = asyncio.run(tracker.flush(settings, send_func=_fake_send))
    assert r2 is True

    # 验证第二次发送的是 PROCESSING 而非 STARTED
    assert len(sent_payloads) == 2
    assert sent_payloads[0].status == "STARTED"
    assert sent_payloads[1].status == "PROCESSING"


def test_callback_tracker_flush_final_retries() -> None:
    """
    测试目的：验证 flush_final 使用完整重试策略发送终态回调。
    预期结果：第 2 次尝试成功时返回 True，总调用 2 次。
    """
    settings = _build_settings()
    call_count = {"n": 0}
    slept: list[float] = []

    async def _fail_then_succeed(
            _payload: KnowledgeImportCallbackPayload,
            _settings: RabbitMQSettings,
    ) -> CallbackAttemptResult:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return CallbackAttemptResult(
                success=False, status_code=500, body_snippet="ERR", error_message=None
            )
        return CallbackAttemptResult(
            success=True, status_code=200, body_snippet="SUCCESS", error_message=None
        )

    async def _fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    tracker = CallbackTracker(task_uuid="test-final")
    tracker.advance(_build_callback_payload("COMPLETED"))
    ok = asyncio.run(
        tracker.flush_final(
            settings,
            max_retries=2,
            retry_delays_seconds=(5,),
            send_func=_fail_then_succeed,
            sleep_func=_fake_sleep,
        )
    )

    assert ok is True
    assert call_count["n"] == 2
    assert slept == [5.0]


# ---------------------------------------------------------------------------
# _handle_incoming_message (集成)
# ---------------------------------------------------------------------------


def test_handle_incoming_message_acks_when_callback_retries_exhausted(
        monkeypatch,
) -> None:
    """
    测试目的：验证回调重试耗尽后消息仍会 ACK，避免队列无限积压。
    预期结果：_handle_incoming_message 执行完成后 fake incoming 的 ack 被调用一次。
    """
    settings = _build_settings()
    message = _build_message()
    payload = _build_callback_payload("FAILED")
    called = {"process": 0}

    class _FakeIncoming:
        """模拟 aio-pika 入站消息对象。"""

        def __init__(self) -> None:
            self.body = message.model_dump_json().encode("utf-8")
            self.ack_called = 0

        async def ack(self) -> None:
            self.ack_called += 1

    async def _fake_process_with_retry(*_args, **_kwargs) -> KnowledgeImportCallbackPayload:
        called["process"] += 1
        return payload

    monkeypatch.setattr(
        "app.core.mq.consumer.process_import_message_with_retry",
        _fake_process_with_retry,
    )

    # Patch _send_callback_once to always fail (tests ACK regardless)
    async def _always_fail_callback(
            _payload: KnowledgeImportCallbackPayload,
            _settings: RabbitMQSettings,
    ) -> CallbackAttemptResult:
        return CallbackAttemptResult(
            success=False, status_code=502, body_snippet="ERR", error_message=None
        )

    monkeypatch.setattr(
        "app.core.mq.consumer._send_callback_once",
        _always_fail_callback,
    )

    incoming = _FakeIncoming()
    asyncio.run(_handle_incoming_message(incoming, settings))

    assert called["process"] == 1
    assert incoming.ack_called == 1
