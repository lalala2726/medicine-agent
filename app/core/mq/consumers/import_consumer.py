from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from app.core.mq._aio_pika_loader import load_aio_pika_consumer
from app.core.mq.config.import_settings import ImportRabbitMQSettings, get_import_settings
from app.core.mq.contracts.import_models import (
    ImportResultStage,
    KnowledgeImportCommandMessage,
    KnowledgeImportResultMessage,
    ProcessingStageDetail,
)
from app.core.mq.observability.import_logger import ImportStage, import_log
from app.core.mq.producers.import_result_publisher import publish_import_result
from app.core.mq.state.import_version_store import is_stale
from app.schemas.knowledge_import import ImportSingleFileSuccessResult
from app.services.knowledge_base_service import import_single_file


def parse_import_command(body: bytes) -> KnowledgeImportCommandMessage:
    """将命令消息字节解析为导入命令模型。

    Args:
        body: MQ 原始消息体字节。

    Returns:
        KnowledgeImportCommandMessage: 解析后的命令对象。

    Raises:
        json.JSONDecodeError: 消息体不是合法 JSON 时抛出。
        ValidationError: 字段校验失败时抛出。
    """
    payload = json.loads(body.decode("utf-8"))
    return KnowledgeImportCommandMessage.model_validate(payload)


def _processing_message(detail: ProcessingStageDetail) -> str:
    """构造处理阶段的用户可读消息。

    Args:
        detail: 处理子阶段枚举。

    Returns:
        str: 对应的阶段消息文本。
    """
    return f"处理中: {detail.value}"


async def _publish_result_event(event: KnowledgeImportResultMessage) -> None:
    """发布单条结果事件并记录结构化日志。

    Args:
        event: 待发布的结果事件。

    Returns:
        None。
    """
    published = await publish_import_result(event)
    if published:
        import_log(
            ImportStage.RESULT_PUBLISHED,
            event.task_uuid,
            biz_key=event.biz_key,
            version=event.version,
            stage=event.stage.value,
        )
        return
    import_log(
        ImportStage.RESULT_PUBLISH_FAILED,
        event.task_uuid,
        biz_key=event.biz_key,
        version=event.version,
        stage=event.stage.value,
    )


async def _emit_started(
        command: KnowledgeImportCommandMessage,
        *,
        started_at: datetime,
) -> None:
    """发送 `STARTED` 结果事件。

    Args:
        command: 导入命令消息。
        started_at: 任务开始时间。

    Returns:
        None。
    """
    event = KnowledgeImportResultMessage.build(
        task_uuid=command.task_uuid,
        biz_key=command.biz_key,
        version=command.version,
        stage=ImportResultStage.STARTED,
        message="任务已接收，即将开始处理",
        knowledge_name=command.knowledge_name,
        document_id=command.document_id,
        file_url=command.file_url,
        embedding_model=command.embedding_model,
        started_at=started_at,
    )
    await _publish_result_event(event)


async def _emit_processing(
        command: KnowledgeImportCommandMessage,
        *,
        started_at: datetime,
        detail: ProcessingStageDetail,
) -> None:
    """发送 `PROCESSING` 结果事件。

    Args:
        command: 导入命令消息。
        started_at: 任务开始时间。
        detail: 当前处理子阶段。

    Returns:
        None。
    """
    event = KnowledgeImportResultMessage.build(
        task_uuid=command.task_uuid,
        biz_key=command.biz_key,
        version=command.version,
        stage=ImportResultStage.PROCESSING,
        stage_detail=detail,
        message=_processing_message(detail),
        knowledge_name=command.knowledge_name,
        document_id=command.document_id,
        file_url=command.file_url,
        embedding_model=command.embedding_model,
        started_at=started_at,
    )
    await _publish_result_event(event)


async def _emit_completed(
        command: KnowledgeImportCommandMessage,
        *,
        started_at: datetime,
        result: ImportSingleFileSuccessResult,
) -> None:
    """发送 `COMPLETED` 结果事件。

    Args:
        command: 导入命令消息。
        started_at: 任务开始时间。
        result: 单文件导入成功结果。

    Returns:
        None。
    """
    event = KnowledgeImportResultMessage.build(
        task_uuid=command.task_uuid,
        biz_key=command.biz_key,
        version=command.version,
        stage=ImportResultStage.COMPLETED,
        message=f"导入成功，chunk_count={result.chunk_count}, vector_count={result.vector_count}",
        knowledge_name=command.knowledge_name,
        document_id=command.document_id,
        file_url=command.file_url,
        embedding_model=command.embedding_model,
        filename=result.filename,
        chunk_count=result.chunk_count,
        vector_count=result.vector_count,
        embedding_dim=result.embedding_dim,
        started_at=started_at,
    )
    await _publish_result_event(event)


async def _emit_failed(
        command: KnowledgeImportCommandMessage,
        *,
        started_at: datetime,
        error_message: str,
        embedding_dim: int = 0,
) -> None:
    """发送 `FAILED` 结果事件。

    Args:
        command: 导入命令消息。
        started_at: 任务开始时间。
        error_message: 失败原因。
        embedding_dim: 已知的向量维度，未知时为 0。

    Returns:
        None。
    """
    event = KnowledgeImportResultMessage.build(
        task_uuid=command.task_uuid,
        biz_key=command.biz_key,
        version=command.version,
        stage=ImportResultStage.FAILED,
        message=error_message,
        knowledge_name=command.knowledge_name,
        document_id=command.document_id,
        file_url=command.file_url,
        embedding_model=command.embedding_model,
        embedding_dim=max(0, embedding_dim),
        started_at=started_at,
    )
    await _publish_result_event(event)


async def _handle_incoming_message(incoming: Any, settings: ImportRabbitMQSettings) -> None:
    """处理单条入站导入命令消息。

    Args:
        incoming: aio-pika 入站消息对象。
        settings: MQ 运行配置。

    Returns:
        None。
    """
    acked = False

    async def _ack_once() -> None:
        """确保同一条入站消息最多 ACK 一次。

        Returns:
            None。
        """
        nonlocal acked
        if acked:
            return
        await incoming.ack()
        acked = True

    raw_message = incoming.body.decode("utf-8", errors="replace")
    print(f"接收到业务端：{raw_message}")

    try:
        command = parse_import_command(incoming.body)
    except (ValidationError, json.JSONDecodeError) as exc:
        import_log(ImportStage.TASK_INVALID, "-", error=str(exc))
        await _ack_once()
        return

    import_log(
        ImportStage.TASK_RECEIVED,
        command.task_uuid,
        biz_key=command.biz_key,
        version=command.version,
        file_url=command.file_url,
    )

    started_at = datetime.now(timezone.utc)

    try:
        if is_stale(
                biz_key=command.biz_key,
                version=command.version,
                settings=settings,
        ):
            import_log(
                ImportStage.TASK_STALE_DROPPED,
                command.task_uuid,
                biz_key=command.biz_key,
                version=command.version,
            )
            await _ack_once()
            return

        await _emit_started(command, started_at=started_at)

        loop = asyncio.get_running_loop()
        stage_queue: asyncio.Queue[ProcessingStageDetail | None] = asyncio.Queue()

        def _on_processing_stage(detail: ProcessingStageDetail) -> None:
            """将线程中的处理阶段事件安全投递到异步队列。

            Args:
                detail: 当前处理子阶段。

            Returns:
                None。
            """
            loop.call_soon_threadsafe(stage_queue.put_nowait, detail)

        async def _drain_processing_events() -> None:
            """持续消费处理阶段事件并转发为 MQ 结果消息。

            Returns:
                None: 收到 ``None`` 哨兵值后结束。
            """
            while True:
                detail = await stage_queue.get()
                if detail is None:
                    return
                await _emit_processing(command, started_at=started_at, detail=detail)

        drain_task = asyncio.create_task(_drain_processing_events())
        result = await asyncio.to_thread(
            import_single_file,
            url=command.file_url,
            knowledge_name=command.knowledge_name,
            document_id=command.document_id,
            embedding_model=command.embedding_model,
            chunk_strategy=command.chunk_strategy,
            chunk_size=command.chunk_size,
            token_size=command.token_size,
            task_uuid=command.task_uuid,
            on_processing_stage=_on_processing_stage,
        )
        loop.call_soon_threadsafe(stage_queue.put_nowait, None)
        await drain_task

        if result.status == "success":
            await _emit_completed(command, started_at=started_at, result=result)
        else:
            import_log(
                ImportStage.FAILED,
                command.task_uuid,
                biz_key=command.biz_key,
                version=command.version,
                error=result.error,
            )
            await _emit_failed(
                command,
                started_at=started_at,
                error_message=result.error,
                embedding_dim=result.embedding_dim,
            )

    except Exception as exc:
        import_log(
            ImportStage.FAILED,
            command.task_uuid,
            biz_key=command.biz_key,
            version=command.version,
            error=str(exc),
        )
        await _emit_failed(command, started_at=started_at, error_message=str(exc))
    finally:
        await _ack_once()


async def _consume_once(settings: ImportRabbitMQSettings) -> None:
    """持续消费导入命令队列直到连接中断或任务取消。

    Args:
        settings: MQ 运行配置。

    Returns:
        None。
    """
    connect_robust, exchange_type_enum = load_aio_pika_consumer()
    connection = await connect_robust(settings.url)
    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=settings.prefetch_count)
        exchange = await channel.declare_exchange(
            settings.exchange,
            exchange_type_enum.DIRECT,
            durable=True,
        )
        queue = await channel.declare_queue(settings.command_queue, durable=True)
        await queue.bind(exchange, routing_key=settings.command_routing_key)
        import_log(
            ImportStage.CONSUMER_CONNECTED,
            "-",
            queue=settings.command_queue,
            routing_key=settings.command_routing_key,
        )
        async with queue.iterator() as queue_iter:
            async for incoming in queue_iter:
                await _handle_incoming_message(incoming, settings)


async def run_import_consumer() -> None:
    """启动导入命令消费者，并在异常时自动重连。"""
    settings = get_import_settings()
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
