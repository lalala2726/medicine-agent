from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from app.core.mq._aio_pika_loader import load_aio_pika_consumer
from app.core.mq.config.document.chunk_add_settings import (
    ChunkAddRabbitMQSettings,
    get_chunk_add_settings,
)
from app.core.mq.contracts.document.chunk_add_models import (
    ChunkAddResultStage,
    KnowledgeChunkAddCommandMessage,
    KnowledgeChunkAddResultMessage,
)
from app.core.mq.observability.document.chunk_add_logger import (
    ChunkAddStage,
    chunk_add_log,
)
from app.core.mq.producers.document.chunk_add_result_publisher import (
    publish_chunk_add_result,
)
from app.services.document_chunk_service import (
    ChunkAddSuccessResult,
    add_document_chunk,
)


def parse_chunk_add_command(body: bytes) -> KnowledgeChunkAddCommandMessage:
    """将命令消息字节解析为手工新增切片命令模型。

    Args:
        body: MQ 原始消息体字节串。

    Returns:
        KnowledgeChunkAddCommandMessage: 解析后的命令对象。

    Raises:
        json.JSONDecodeError: 消息体不是合法 JSON 时抛出。
        ValidationError: JSON 字段不满足 command 模型约束时抛出。
    """
    payload = json.loads(body.decode("utf-8"))
    return KnowledgeChunkAddCommandMessage.model_validate(payload)


async def _emit_started(
        command: KnowledgeChunkAddCommandMessage,
        *,
        started_at: datetime,
) -> bool:
    """发布手工新增切片 `STARTED` 结果事件。

    Args:
        command: 已通过校验的命令消息。
        started_at: 任务开始时间。

    Returns:
        bool: 结果消息成功投递到 MQ 时返回 ``True``。
    """
    event = KnowledgeChunkAddResultMessage.build(
        task_uuid=command.task_uuid,
        chunk_id=command.chunk_id,
        stage=ChunkAddResultStage.STARTED,
        message="任务已接收，即将开始处理",
        knowledge_name=command.knowledge_name,
        document_id=command.document_id,
        embedding_model=command.embedding_model,
        started_at=started_at,
    )
    return await publish_chunk_add_result(event)


async def _emit_completed(
        command: KnowledgeChunkAddCommandMessage,
        *,
        started_at: datetime,
        result: ChunkAddSuccessResult,
) -> bool:
    """发布手工新增切片 `COMPLETED` 结果事件。

    Args:
        command: 已通过校验的命令消息。
        started_at: 任务开始时间。
        result: 手工新增切片成功结果。

    Returns:
        bool: 结果消息成功投递到 MQ 时返回 ``True``。
    """
    event = KnowledgeChunkAddResultMessage.build(
        task_uuid=command.task_uuid,
        chunk_id=command.chunk_id,
        stage=ChunkAddResultStage.COMPLETED,
        message="切片新增成功",
        knowledge_name=command.knowledge_name,
        document_id=command.document_id,
        embedding_model=command.embedding_model,
        vector_id=result.vector_id,
        chunk_index=result.chunk_index,
        embedding_dim=result.embedding_dim,
        started_at=started_at,
    )
    return await publish_chunk_add_result(event)


async def _emit_failed(
        command: KnowledgeChunkAddCommandMessage,
        *,
        started_at: datetime,
        error_message: str,
        embedding_dim: int = 0,
) -> bool:
    """发布手工新增切片 `FAILED` 结果事件。

    Args:
        command: 已通过校验的命令消息。
        started_at: 任务开始时间。
        error_message: 失败原因。
        embedding_dim: 已知向量维度，未知时为 ``0``。

    Returns:
        bool: 结果消息成功投递到 MQ 时返回 ``True``。
    """
    event = KnowledgeChunkAddResultMessage.build(
        task_uuid=command.task_uuid,
        chunk_id=command.chunk_id,
        stage=ChunkAddResultStage.FAILED,
        message=error_message,
        knowledge_name=command.knowledge_name,
        document_id=command.document_id,
        embedding_model=command.embedding_model,
        embedding_dim=max(0, embedding_dim),
        started_at=started_at,
    )
    return await publish_chunk_add_result(event)


async def _handle_incoming_message(
        incoming: Any,
        settings: ChunkAddRabbitMQSettings,
) -> None:
    """处理单条入站手工新增切片命令消息。

    Args:
        incoming: aio-pika 入站消息对象。
        settings: 当前消费者使用的 MQ 配置。

    Returns:
        None: 无返回值；通过 ACK 和结果消息驱动后续流程。
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

    try:
        command = parse_chunk_add_command(incoming.body)
    except (ValidationError, json.JSONDecodeError) as exc:
        chunk_add_log(
            ChunkAddStage.TASK_INVALID,
            error=exc,
            queue=settings.command_queue,
            body=incoming.body.decode("utf-8", errors="replace"),
        )
        await _ack_once()
        return

    chunk_add_log(
        ChunkAddStage.TASK_RECEIVED,
        command.task_uuid,
        chunk_id=command.chunk_id,
        knowledge_name=command.knowledge_name,
        document_id=command.document_id,
        embedding_model=command.embedding_model,
    )

    started_at = datetime.now(timezone.utc)

    started_published = await _emit_started(command, started_at=started_at)
    if not started_published:
        chunk_add_log(
            ChunkAddStage.RESULT_PUBLISH_FAILED,
            command.task_uuid,
            stage="STARTED",
            queue=settings.command_queue,
        )
        return
    chunk_add_log(
        ChunkAddStage.RESULT_PUBLISHED,
        command.task_uuid,
        stage="STARTED",
        chunk_id=command.chunk_id,
    )

    try:
        result = await asyncio.to_thread(
            add_document_chunk,
            knowledge_name=command.knowledge_name,
            document_id=command.document_id,
            content=command.content,
            embedding_model=command.embedding_model,
        )
    except Exception as exc:
        chunk_add_log(
            ChunkAddStage.ADD_FAILED,
            command.task_uuid,
            chunk_id=command.chunk_id,
            error=str(exc),
        )
        failed_published = await _emit_failed(
            command,
            started_at=started_at,
            error_message=str(exc),
        )
        if failed_published:
            chunk_add_log(
                ChunkAddStage.RESULT_PUBLISHED,
                command.task_uuid,
                stage="FAILED",
                chunk_id=command.chunk_id,
            )
            await _ack_once()
            return
        chunk_add_log(
            ChunkAddStage.RESULT_PUBLISH_FAILED,
            command.task_uuid,
            stage="FAILED",
            queue=settings.command_queue,
        )
        return

    completed_published = await _emit_completed(
        command,
        started_at=started_at,
        result=result,
    )
    if completed_published:
        chunk_add_log(
            ChunkAddStage.COMPLETED,
            command.task_uuid,
            chunk_id=command.chunk_id,
            vector_id=result.vector_id,
            chunk_index=result.chunk_index,
            embedding_dim=result.embedding_dim,
        )
        chunk_add_log(
            ChunkAddStage.RESULT_PUBLISHED,
            command.task_uuid,
            stage="COMPLETED",
            chunk_id=command.chunk_id,
        )
        await _ack_once()
        return

    chunk_add_log(
        ChunkAddStage.RESULT_PUBLISH_FAILED,
        command.task_uuid,
        stage="COMPLETED",
        queue=settings.command_queue,
    )


async def _consume_once(settings: ChunkAddRabbitMQSettings) -> None:
    """持续消费手工新增切片命令队列直到连接中断或任务取消。

    Args:
        settings: 当前消费者使用的 MQ 配置。

    Returns:
        None: 队列消费循环持续运行，直到连接中断或任务取消。
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
        chunk_add_log(
            ChunkAddStage.CONSUMER_CONNECTED,
            queue=settings.command_queue,
            routing_key=settings.command_routing_key,
        )
        async with queue.iterator() as queue_iter:
            async for incoming in queue_iter:
                await _handle_incoming_message(incoming, settings)


async def run_chunk_add_consumer() -> None:
    """启动手工新增切片命令消费者，并在异常时自动重连。

    Returns:
        None: 常驻协程，除非任务被取消否则不会主动退出。
    """
    settings = get_chunk_add_settings()
    while True:
        try:
            await _consume_once(settings)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            chunk_add_log(
                ChunkAddStage.CONSUMER_RECONNECTING,
                error=str(exc),
                retry_after_seconds=5,
            )
            await asyncio.sleep(5)
