from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException
from app.core.mq.import_logger import ImportStage, import_log
from app.core.mq.models import (
    CallbackStage,
    KnowledgeImportCallbackPayload,
    KnowledgeImportMessage,
)
from app.core.mq.settings import RabbitMQSettings, get_rabbitmq_settings
from app.services.knowledge_base_service import import_single_file
from app.utils.http_client import HttpClient


# ---------------------------------------------------------------------------
# CallbackAttemptResult
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CallbackAttemptResult:
    """
    功能描述:
        定义单次回调请求执行结果，供回调重试流程记录可观测信息。

    参数说明:
        success (bool): 单次回调是否满足成功条件（HTTP 200 且响应体为 SUCCESS）。
        status_code (int | None): 回调响应状态码，请求异常时为 None。
        body_snippet (str): 响应体截断摘要，避免日志写入超长内容。
        error_message (str | None): 请求异常或失败原因摘要，成功时为 None。

    返回值:
        无。该类用于承载一次回调尝试结果。

    异常说明:
        无。
    """

    success: bool
    status_code: int | None
    body_snippet: str
    error_message: str | None


# ---------------------------------------------------------------------------
# CallbackTracker — 多阶段回调状态追踪器
# ---------------------------------------------------------------------------


@dataclass
class CallbackTracker:
    """
    功能描述:
        维护单个导入任务的最新回调状态，实现「只发最新」的合并语义。

        - 中间阶段（STARTED / PROCESSING）回调失败时不阻塞流程，仅记日志。
        - 终态阶段（COMPLETED / FAILED）使用完整重试策略。
        - 如果前一个阶段回调尚未成功、新的阶段已到达，flush 时只发当前最新状态。

    参数说明:
        task_uuid (str): 导入任务唯一标识。
        _current_payload (KnowledgeImportCallbackPayload | None): 当前最新待发的回调 payload。
        _pending (bool): 是否有待发送的新状态。
    """

    task_uuid: str
    _current_payload: KnowledgeImportCallbackPayload | None = field(
        default=None, repr=False
    )
    _pending: bool = field(default=False, repr=False)

    def advance(
        self, payload: KnowledgeImportCallbackPayload
    ) -> None:
        """
        功能描述:
            推进回调状态到新阶段，标记为待发送。已有的旧 payload 会被新的覆盖。

        参数说明:
            payload (KnowledgeImportCallbackPayload): 新阶段的回调数据。

        返回值:
            None。

        异常说明:
            无。
        """
        self._current_payload = payload
        self._pending = True

    async def flush(
        self,
        settings: RabbitMQSettings,
        *,
        send_func: Callable[
            [KnowledgeImportCallbackPayload, RabbitMQSettings],
            Awaitable[CallbackAttemptResult],
        ] = None,
    ) -> bool:
        """
        功能描述:
            尝试发送当前最新的中间阶段回调（单次，不重试）。
            失败不阻塞流程，只记录日志。

        参数说明:
            settings (RabbitMQSettings): MQ 与回调配置。
            send_func: 可选覆盖发送函数（测试用）。

        返回值:
            bool: 发送成功返回 True，无待发或失败返回 False。

        异常说明:
            无。异常在内部捕获并记录日志。
        """
        if not self._pending or self._current_payload is None:
            return False

        actual_send = send_func or _send_callback_once
        result = await actual_send(self._current_payload, settings)
        stage = self._current_payload.status

        if result.success:
            import_log(
                ImportStage.CALLBACK_SENT,
                self.task_uuid,
                stage=stage,
                status_code=result.status_code,
            )
            self._pending = False
            return True

        import_log(
            ImportStage.CALLBACK_FAILED,
            self.task_uuid,
            stage=stage,
            status_code=result.status_code,
            error=result.error_message or result.body_snippet,
            action="continue",
        )
        # 中间阶段失败不阻塞，标记已尝试
        self._pending = False
        return False

    async def flush_final(
        self,
        settings: RabbitMQSettings,
        *,
        max_retries: int,
        retry_delays_seconds: tuple[int, ...],
        send_func: Callable[
            [KnowledgeImportCallbackPayload, RabbitMQSettings],
            Awaitable[CallbackAttemptResult],
        ] = None,
        sleep_func: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> bool:
        """
        功能描述:
            发送终态回调（COMPLETED / FAILED），使用完整重试策略。

        参数说明:
            settings (RabbitMQSettings): MQ 与回调配置。
            max_retries (int): 最大重试次数（不含首次执行）。
            retry_delays_seconds (tuple[int, ...]): 重试间隔秒数。
            send_func: 可选覆盖发送函数（测试用）。
            sleep_func: 可选覆盖休眠函数（测试用）。

        返回值:
            bool: 窗口内成功返回 True，超限失败返回 False。

        异常说明:
            无。
        """
        if self._current_payload is None:
            return False

        actual_send = send_func or _send_callback_once
        ok = await send_callback_with_retry(
            self._current_payload,
            settings=settings,
            max_retries=max_retries,
            retry_delays_seconds=retry_delays_seconds,
            send_func=actual_send,
            sleep_func=sleep_func,
        )

        stage = self._current_payload.status
        if ok:
            import_log(ImportStage.CALLBACK_SENT, self.task_uuid, stage=stage)
        else:
            import_log(ImportStage.CALLBACK_FAILED, self.task_uuid, stage=stage, action="exhausted")

        self._pending = False
        return ok

    @property
    def current_payload(self) -> KnowledgeImportCallbackPayload | None:
        """当前最新 payload（只读）。"""
        return self._current_payload


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _build_body_snippet(body_text: str, *, max_length: int = 200) -> str:
    """
    功能描述:
        生成响应体日志摘要，限制长度避免污染日志与告警系统。

    参数说明:
        body_text (str): 原始响应体文本。
        max_length (int): 摘要最大字符长度，默认值为 200。

    返回值:
        str: 处理后的响应体摘要文本。

    异常说明:
        无。
    """
    normalized = body_text.strip()
    if len(normalized) <= max_length:
        return normalized
    return f"{normalized[:max_length]}..."


def _load_aio_pika() -> tuple[Any, Any]:
    """
    功能描述:
        懒加载 aio-pika 依赖，避免在模块导入阶段产生硬依赖。

    参数说明:
        无。

    返回值:
        tuple[Any, Any]:
            - connect_robust 函数
            - ExchangeType 枚举

    异常说明:
        ServiceException: 未安装 aio-pika 依赖时抛出。
    """
    try:
        from aio_pika import ExchangeType, connect_robust
    except Exception as exc:
        raise ServiceException(
            code=ResponseCode.INTERNAL_ERROR,
            message="缺少 aio-pika 依赖，无法启动 MQ 消费者",
        ) from exc
    return connect_robust, ExchangeType


def _build_failed_payload(
    message: KnowledgeImportMessage,
    *,
    started_at: datetime,
    error_message: str,
    embedding_dim: int = 0,
    stage_detail: str | None = None,
) -> KnowledgeImportCallbackPayload:
    """
    功能描述:
        构造失败回调参数对象，供异常与重试失败场景复用。

    参数说明:
        message (KnowledgeImportMessage): 导入消息。
        started_at (datetime): 处理开始时间。
        error_message (str): 失败原因。
        embedding_dim (int): 向量维度，未知时可传 0，默认值为 0。
        stage_detail (str | None): 可选子阶段描述。

    返回值:
        KnowledgeImportCallbackPayload: 失败状态回调参数。

    异常说明:
        无。
    """
    return KnowledgeImportCallbackPayload.build(
        task_uuid=message.task_uuid,
        knowledge_name=message.knowledge_name,
        document_id=message.document_id,
        file_url=message.file_url,
        status=CallbackStage.FAILED.value,
        message=error_message,
        embedding_model=message.embedding_model,
        embedding_dim=max(0, embedding_dim),
        chunk_strategy=message.chunk_strategy.value,
        chunk_size=message.chunk_size,
        token_size=message.token_size,
        chunk_count=0,
        vector_count=0,
        started_at=started_at,
        finished_at=datetime.now(timezone.utc),
        stage_detail=stage_detail,
    )


def _build_base_payload(
    message: KnowledgeImportMessage,
    *,
    status: str,
    detail: str,
    started_at: datetime,
) -> KnowledgeImportCallbackPayload:
    """
    功能描述:
        构造通用阶段回调 payload（STARTED / PROCESSING 等）。

    参数说明:
        message (KnowledgeImportMessage): 导入消息。
        status (str): 回调阶段。
        detail (str): 状态描述。
        started_at (datetime): 处理开始时间。

    返回值:
        KnowledgeImportCallbackPayload: 阶段回调参数。

    异常说明:
        无。
    """
    return KnowledgeImportCallbackPayload.build(
        task_uuid=message.task_uuid,
        knowledge_name=message.knowledge_name,
        document_id=message.document_id,
        file_url=message.file_url,
        status=status,
        message=detail,
        embedding_model=message.embedding_model,
        embedding_dim=0,
        chunk_strategy=message.chunk_strategy.value,
        chunk_size=message.chunk_size,
        token_size=message.token_size,
        chunk_count=0,
        vector_count=0,
        started_at=started_at,
        finished_at=datetime.now(timezone.utc),
        stage_detail=detail,
    )


# ---------------------------------------------------------------------------
# 消息处理 — 单次执行 + 重试包装
# ---------------------------------------------------------------------------


def process_import_message_once(
    message: KnowledgeImportMessage,
    *,
    tracker: CallbackTracker | None = None,
) -> KnowledgeImportCallbackPayload:
    """
    功能描述:
        执行单条导入消息处理（下载、解析、切片、向量化、入库），并构造回调参数。

    参数说明:
        message (KnowledgeImportMessage): 导入任务消息。
        tracker (CallbackTracker | None): 可选回调追踪器，用于中间阶段回调。

    返回值:
        KnowledgeImportCallbackPayload: 本次处理对应的回调参数对象。

    异常说明:
        无。函数内部捕获异常并转换为失败回调对象。
    """
    started_at = datetime.now(timezone.utc)
    try:
        result = import_single_file(
            url=message.file_url,
            knowledge_name=message.knowledge_name,
            document_id=message.document_id,
            embedding_model=message.embedding_model,
            chunk_strategy=message.chunk_strategy,
            chunk_size=message.chunk_size,
            token_size=message.token_size,
            task_uuid=message.task_uuid,
        )
    except Exception as exc:
        import_log(
            ImportStage.FAILED,
            message.task_uuid,
            error=str(exc),
        )
        return _build_failed_payload(
            message,
            started_at=started_at,
            error_message=str(exc),
        )

    if result.get("status") == "success":
        return KnowledgeImportCallbackPayload.build(
            task_uuid=message.task_uuid,
            knowledge_name=message.knowledge_name,
            document_id=message.document_id,
            file_url=message.file_url,
            status=CallbackStage.COMPLETED.value,
            message=(
                f"导入成功，chunk_count={result.get('chunk_count', 0)}, "
                f"vector_count={result.get('vector_count', 0)}"
            ),
            embedding_model=message.embedding_model,
            embedding_dim=int(result.get("embedding_dim") or 0),
            chunk_strategy=message.chunk_strategy.value,
            chunk_size=message.chunk_size,
            token_size=message.token_size,
            chunk_count=int(result.get("chunk_count") or 0),
            vector_count=int(result.get("vector_count") or 0),
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
        )

    error_msg = result.get("error") or "导入失败"
    return _build_failed_payload(
        message,
        started_at=started_at,
        error_message=str(error_msg),
        embedding_dim=int(result.get("embedding_dim") or 0),
    )


async def process_import_message_with_retry(
    message: KnowledgeImportMessage,
    *,
    max_retries: int,
    retry_delays_seconds: tuple[int, ...],
    tracker: CallbackTracker | None = None,
    process_func: Callable[
        ..., KnowledgeImportCallbackPayload
    ] = None,
    sleep_func: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> KnowledgeImportCallbackPayload:
    """
    功能描述:
        对单条导入消息执行带重试处理，重试耗尽后返回最后一次处理结果。

    参数说明:
        message (KnowledgeImportMessage): 导入任务消息。
        max_retries (int): 最大重试次数（不含首次执行）。
        retry_delays_seconds (tuple[int, ...]): 重试等待秒数列表。
        tracker (CallbackTracker | None): 可选回调追踪器。
        process_func: 实际处理函数，默认值为 process_import_message_once。
        sleep_func (Callable[[float], Awaitable[None]]): 异步休眠函数。

    返回值:
        KnowledgeImportCallbackPayload: 最终处理结果（成功或失败）。

    异常说明:
        无。异常会被捕获并转为失败回调对象。
    """
    if process_func is None:
        process_func = process_import_message_once

    total_attempts = max_retries + 1
    last_payload: KnowledgeImportCallbackPayload | None = None
    for attempt in range(total_attempts):
        try:
            payload = process_func(message, tracker=tracker)
            last_payload = payload
            if payload.status == CallbackStage.COMPLETED.value:
                return payload
        except Exception as exc:
            last_payload = _build_failed_payload(
                message,
                started_at=datetime.now(timezone.utc),
                error_message=str(exc),
            )

        if attempt >= total_attempts - 1:
            break

        delay_index = min(attempt, max(0, len(retry_delays_seconds) - 1))
        delay = float(retry_delays_seconds[delay_index])
        import_log(
            ImportStage.RETRY_SCHEDULED,
            message.task_uuid,
            attempt=attempt + 1,
            max_retries=max_retries,
            delay_seconds=delay,
        )
        await sleep_func(delay)

    return last_payload or _build_failed_payload(
        message,
        started_at=datetime.now(timezone.utc),
        error_message="导入失败且未产生结果",
    )


# ---------------------------------------------------------------------------
# 消息解析
# ---------------------------------------------------------------------------


def parse_import_message(body: bytes) -> KnowledgeImportMessage:
    """
    功能描述:
        将 RabbitMQ 原始消息体解析为导入任务模型。

    参数说明:
        body (bytes): MQ 消息体字节串。

    返回值:
        KnowledgeImportMessage: 解析后的导入任务消息对象。

    异常说明:
        ValidationError: JSON 结构非法或字段校验失败时抛出。
        JSONDecodeError: 消息体不是合法 JSON 时抛出。
    """
    payload = json.loads(body.decode("utf-8"))
    return KnowledgeImportMessage.model_validate(payload)


# ---------------------------------------------------------------------------
# 回调发送
# ---------------------------------------------------------------------------


def _is_callback_success(status_code: int, body_text: str) -> bool:
    """
    功能描述:
        判断回调请求是否成功（必须满足 HTTP 200 且响应体严格等于 SUCCESS）。

    参数说明:
        status_code (int): HTTP 状态码。
        body_text (str): 响应体文本。

    返回值:
        bool: 满足成功规则返回 True，否则返回 False。

    异常说明:
        无。
    """
    return status_code == 200 and body_text.strip() == "SUCCESS"


async def _send_callback_once(
    payload: KnowledgeImportCallbackPayload,
    settings: RabbitMQSettings,
) -> CallbackAttemptResult:
    """
    功能描述:
        执行一次导入结果回调请求（POST + JSON body）。

    参数说明:
        payload (KnowledgeImportCallbackPayload): 回调参数对象。
        settings (RabbitMQSettings): MQ 与回调配置。

    返回值:
        CallbackAttemptResult: 单次回调执行结果。

    异常说明:
        无。异常在函数内部捕获并转换为失败结果对象。
    """
    try:
        async with HttpClient(timeout=float(settings.callback_timeout_seconds)) as client:
            response = await client.post(
                settings.callback_url,
                json=payload.to_callback_body(),
                response_format="raw",
            )
        response_text = response.text if response.text is not None else ""
        return CallbackAttemptResult(
            success=_is_callback_success(response.status_code, response_text),
            status_code=response.status_code,
            body_snippet=_build_body_snippet(response_text),
            error_message=None,
        )
    except Exception as exc:
        return CallbackAttemptResult(
            success=False,
            status_code=None,
            body_snippet="",
            error_message=str(exc),
        )


async def send_callback_with_retry(
    payload: KnowledgeImportCallbackPayload,
    *,
    settings: RabbitMQSettings,
    max_retries: int,
    retry_delays_seconds: tuple[int, ...],
    send_func: Callable[
        [KnowledgeImportCallbackPayload, RabbitMQSettings], Awaitable[CallbackAttemptResult]
    ] = _send_callback_once,
    sleep_func: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> bool:
    """
    功能描述:
        对回调请求执行重试，直到成功或超过重试窗口。

    参数说明:
        payload (KnowledgeImportCallbackPayload): 回调参数对象。
        settings (RabbitMQSettings): MQ 与回调配置。
        max_retries (int): 最大重试次数（不含首次执行）。
        retry_delays_seconds (tuple[int, ...]): 重试间隔（秒）。
        send_func: 单次回调发送函数。
        sleep_func: 异步休眠函数。

    返回值:
        bool: 回调在窗口内成功返回 True，超时失败返回 False。

    异常说明:
        无。异常在内部处理并按失败重试。
    """
    total_attempts = max_retries + 1
    for attempt in range(total_attempts):
        attempt_result = await send_func(payload, settings)
        next_delay_seconds: float | None = None
        if not attempt_result.success and attempt < total_attempts - 1:
            delay_index = min(attempt, max(0, len(retry_delays_seconds) - 1))
            next_delay_seconds = float(retry_delays_seconds[delay_index])
        if attempt_result.success:
            return True
        if next_delay_seconds is None:
            break
        await sleep_func(next_delay_seconds)
    return False


# ---------------------------------------------------------------------------
# 消息入口 — 消费 → 处理 → 回调 → ACK
# ---------------------------------------------------------------------------


async def _handle_incoming_message(
    incoming: Any,
    settings: RabbitMQSettings,
) -> None:
    """
    功能描述:
        处理一条 RabbitMQ 消息：解析 → STARTED 回调 → 处理（含重试）→ 终态回调 → ACK。

    参数说明:
        incoming (IncomingMessage): RabbitMQ 入站消息对象。
        settings (RabbitMQSettings): RabbitMQ 运行配置。

    返回值:
        None。

    异常说明:
        无。异常会在函数内部记录并最终 ACK。
    """
    # Step 1: 解析消息
    try:
        message = parse_import_message(incoming.body)
    except (ValidationError, json.JSONDecodeError) as exc:
        import_log(ImportStage.TASK_INVALID, "-", error=str(exc))
        await incoming.ack()
        return

    task_uuid = message.task_uuid
    import_log(ImportStage.TASK_RECEIVED, task_uuid, file_url=message.file_url)
    started_at = datetime.now(timezone.utc)
    # Step 2: 初始化追踪器并发送 STARTED 回调
    tracker = CallbackTracker(task_uuid=task_uuid)
    started_payload = _build_base_payload(
        message,
        status=CallbackStage.STARTED.value,
        detail="任务已接收，即将开始处理",
        started_at=started_at,
    )
    tracker.advance(started_payload)
    await tracker.flush(settings)

    # Step 3: 处理消息（含重试）
    callback_payload = await process_import_message_with_retry(
        message,
        max_retries=settings.max_retries,
        retry_delays_seconds=settings.retry_delays_seconds,
        tracker=tracker,
    )

    # Step 4: 发送终态回调（含重试）
    tracker.advance(callback_payload)
    await tracker.flush_final(
        settings,
        max_retries=settings.callback_max_retries,
        retry_delays_seconds=settings.callback_retry_delays_seconds,
    )

    # Step 5: ACK
    await incoming.ack()


# ---------------------------------------------------------------------------
# 消费者主循环
# ---------------------------------------------------------------------------


async def _consume_once(settings: RabbitMQSettings) -> None:
    """
    功能描述:
        建立 RabbitMQ 连接并持续消费导入队列消息，直到连接中断或任务取消。

    参数说明:
        settings (RabbitMQSettings): RabbitMQ 运行配置。

    返回值:
        None。

    异常说明:
        Exception: 连接失败或消费失败时抛出，由上层循环处理重连。
    """
    connect_robust, exchange_type_enum = _load_aio_pika()
    connection = await connect_robust(settings.url)
    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=settings.prefetch_count)
        exchange = await channel.declare_exchange(
            settings.exchange,
            exchange_type_enum.DIRECT,
            durable=True,
        )
        queue = await channel.declare_queue(settings.queue, durable=True)
        await queue.bind(exchange, routing_key=settings.routing_key)
        import_log(ImportStage.CONSUMER_CONNECTED, "-", queue=settings.queue)
        async with queue.iterator() as queue_iter:
            async for incoming in queue_iter:
                await _handle_incoming_message(incoming, settings)


async def run_import_consumer() -> None:
    """
    功能描述:
        启动导入消息消费者并保持常驻，消费异常时自动等待后重连。

    参数说明:
        无。配置从环境变量加载。

    返回值:
        None。

    异常说明:
        无。非取消异常会被捕获并重试。
    """
    settings = get_rabbitmq_settings()
    while True:
        try:
            await _consume_once(settings)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            import_log(
                ImportStage.CONSUMER_RECONNECTING,
                "-",
                error=str(exc),
                retry_after_seconds=5,
            )
            await asyncio.sleep(5)
