from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from app.core.mq._aio_pika_loader import load_aio_pika_consumer
from app.core.mq.config.chunk_rebuild_settings import (
    ChunkRebuildRabbitMQSettings,
    get_chunk_rebuild_settings,
)
from app.core.mq.contracts.chunk_rebuild_models import (
    ChunkRebuildResultStage,
    KnowledgeChunkRebuildCommandMessage,
    KnowledgeChunkRebuildResultMessage,
)
from app.core.mq.observability.chunk_rebuild_logger import (
    ChunkRebuildStage,
    chunk_rebuild_log,
)
from app.core.mq.producers.chunk_rebuild_result_publisher import (
    publish_chunk_rebuild_result,
)
from app.core.mq.state.chunk_rebuild_version_store import get_latest_version
from app.services.chunk_service import (
    ChunkRebuildMessageStaleError,
    ChunkRebuildSuccessResult,
    rebuild_document_chunk,
)


def parse_chunk_rebuild_command(body: bytes) -> KnowledgeChunkRebuildCommandMessage:
    """将命令消息字节解析为切片重建命令模型。

    Args:
        body: MQ 原始消息体字节串。

    Returns:
        KnowledgeChunkRebuildCommandMessage: 解析后的命令对象。

    Raises:
        json.JSONDecodeError: 消息体不是合法 JSON 时抛出。
        ValidationError: JSON 字段不满足 command 模型约束时抛出。
    """
    payload = json.loads(body.decode("utf-8"))
    return KnowledgeChunkRebuildCommandMessage.model_validate(payload)


def _build_stale_reason(*, vector_id: int, version: int, latest_version: int) -> str:
    """构造过期任务的统一原因文案。

    Args:
        vector_id: Milvus 向量主键 ID。
        version: 当前消息版本号。
        latest_version: Redis 中记录的最新版本号。

    Returns:
        str: 统一格式的过期原因描述。
    """
    return (
        "切片重建任务已过期，已被更新版本替代，"
        f"vector_id={vector_id}, message_version={version}, latest_version={latest_version}"
    )


async def _emit_started(
        command: KnowledgeChunkRebuildCommandMessage,
        *,
        started_at: datetime,
) -> bool:
    """发布切片重建 `STARTED` 结果事件。

    Args:
        command: 已通过校验的命令消息。
        started_at: 任务开始时间。

    Returns:
        bool: 结果消息成功投递到 MQ 时返回 ``True``。
    """
    event = KnowledgeChunkRebuildResultMessage.build(
        task_uuid=command.task_uuid,
        version=command.version,
        stage=ChunkRebuildResultStage.STARTED,
        message="任务已接收，即将开始处理",
        knowledge_name=command.knowledge_name,
        document_id=command.document_id,
        vector_id=command.vector_id,
        embedding_model=command.embedding_model,
        started_at=started_at,
    )
    return await publish_chunk_rebuild_result(event)


async def _emit_completed(
        command: KnowledgeChunkRebuildCommandMessage,
        *,
        started_at: datetime,
        result: ChunkRebuildSuccessResult,
) -> bool:
    """发布切片重建 `COMPLETED` 结果事件。

    Args:
        command: 已通过校验的命令消息。
        started_at: 任务开始时间。
        result: 切片重建成功结果。

    Returns:
        bool: 结果消息成功投递到 MQ 时返回 ``True``。
    """
    event = KnowledgeChunkRebuildResultMessage.build(
        task_uuid=command.task_uuid,
        version=command.version,
        stage=ChunkRebuildResultStage.COMPLETED,
        message="切片重建成功",
        knowledge_name=command.knowledge_name,
        document_id=command.document_id,
        vector_id=command.vector_id,
        embedding_model=command.embedding_model,
        embedding_dim=result.embedding_dim,
        started_at=started_at,
    )
    return await publish_chunk_rebuild_result(event)


async def _emit_failed(
        command: KnowledgeChunkRebuildCommandMessage,
        *,
        started_at: datetime,
        error_message: str,
        embedding_dim: int = 0,
) -> bool:
    """发布切片重建 `FAILED` 结果事件。

    Args:
        command: 已通过校验的命令消息。
        started_at: 任务开始时间。
        error_message: 失败原因或丢弃原因。
        embedding_dim: 已知向量维度，未知时为 ``0``。

    Returns:
        bool: 结果消息成功投递到 MQ 时返回 ``True``。
    """
    event = KnowledgeChunkRebuildResultMessage.build(
        task_uuid=command.task_uuid,
        version=command.version,
        stage=ChunkRebuildResultStage.FAILED,
        message=error_message,
        knowledge_name=command.knowledge_name,
        document_id=command.document_id,
        vector_id=command.vector_id,
        embedding_model=command.embedding_model,
        embedding_dim=max(0, embedding_dim),
        started_at=started_at,
    )
    return await publish_chunk_rebuild_result(event)


async def _handle_incoming_message(
        incoming: Any,
        settings: ChunkRebuildRabbitMQSettings,
) -> None:
    """处理单条入站切片重建命令消息。

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
        command = parse_chunk_rebuild_command(incoming.body)
    except (ValidationError, json.JSONDecodeError) as exc:
        chunk_rebuild_log(
            ChunkRebuildStage.TASK_INVALID,
            error=exc,
            queue=settings.command_queue,
            body=incoming.body.decode("utf-8", errors="replace"),
        )
        await _ack_once()
        return

    chunk_rebuild_log(
        ChunkRebuildStage.TASK_RECEIVED,
        command.task_uuid,
        version=command.version,
        knowledge_name=command.knowledge_name,
        document_id=command.document_id,
        vector_id=command.vector_id,
        embedding_model=command.embedding_model,
    )

    started_at = datetime.now(timezone.utc)
    latest_version = get_latest_version(
        vector_id=command.vector_id,
        settings=settings,
    )
    if latest_version is not None and command.version < latest_version:
        reason = _build_stale_reason(
            vector_id=command.vector_id,
            version=command.version,
            latest_version=latest_version,
        )
        chunk_rebuild_log(
            ChunkRebuildStage.TASK_STALE_DROPPED,
            command.task_uuid,
            vector_id=command.vector_id,
            version=command.version,
            latest_version=latest_version,
        )
        failed_published = await _emit_failed(
            command,
            started_at=started_at,
            error_message=reason,
        )
        if failed_published:
            chunk_rebuild_log(
                ChunkRebuildStage.RESULT_PUBLISHED,
                command.task_uuid,
                stage="FAILED",
                vector_id=command.vector_id,
                version=command.version,
            )
            await _ack_once()
        else:
            chunk_rebuild_log(
                ChunkRebuildStage.RESULT_PUBLISH_FAILED,
                command.task_uuid,
                stage="FAILED",
                queue=settings.command_queue,
            )
        return

    started_published = await _emit_started(command, started_at=started_at)
    if not started_published:
        chunk_rebuild_log(
            ChunkRebuildStage.RESULT_PUBLISH_FAILED,
            command.task_uuid,
            stage="STARTED",
            queue=settings.command_queue,
        )
        return
    chunk_rebuild_log(
        ChunkRebuildStage.RESULT_PUBLISHED,
        command.task_uuid,
        stage="STARTED",
        vector_id=command.vector_id,
        version=command.version,
    )

    try:
        result = await asyncio.to_thread(
            rebuild_document_chunk,
            knowledge_name=command.knowledge_name,
            document_id=command.document_id,
            vector_id=command.vector_id,
            version=command.version,
            content=command.content,
            embedding_model=command.embedding_model,
        )
    except ChunkRebuildMessageStaleError as exc:
        chunk_rebuild_log(
            ChunkRebuildStage.REBUILD_STALE,
            command.task_uuid,
            vector_id=command.vector_id,
            version=command.version,
            reason=str(exc),
        )
        failed_published = await _emit_failed(
            command,
            started_at=started_at,
            error_message=str(exc),
        )
        if failed_published:
            chunk_rebuild_log(
                ChunkRebuildStage.RESULT_PUBLISHED,
                command.task_uuid,
                stage="FAILED",
                vector_id=command.vector_id,
                version=command.version,
            )
            await _ack_once()
            return
        chunk_rebuild_log(
            ChunkRebuildStage.RESULT_PUBLISH_FAILED,
            command.task_uuid,
            stage="FAILED",
            queue=settings.command_queue,
        )
        return
    except Exception as exc:
        chunk_rebuild_log(
            ChunkRebuildStage.REBUILD_FAILED,
            command.task_uuid,
            vector_id=command.vector_id,
            version=command.version,
            error=str(exc),
        )
        failed_published = await _emit_failed(
            command,
            started_at=started_at,
            error_message=str(exc),
        )
        if failed_published:
            chunk_rebuild_log(
                ChunkRebuildStage.RESULT_PUBLISHED,
                command.task_uuid,
                stage="FAILED",
                vector_id=command.vector_id,
                version=command.version,
            )
            await _ack_once()
            return
        chunk_rebuild_log(
            ChunkRebuildStage.RESULT_PUBLISH_FAILED,
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
        chunk_rebuild_log(
            ChunkRebuildStage.COMPLETED,
            command.task_uuid,
            vector_id=command.vector_id,
            version=command.version,
            embedding_dim=result.embedding_dim,
        )
        chunk_rebuild_log(
            ChunkRebuildStage.RESULT_PUBLISHED,
            command.task_uuid,
            stage="COMPLETED",
            vector_id=command.vector_id,
            version=command.version,
        )
        await _ack_once()
        return

    chunk_rebuild_log(
        ChunkRebuildStage.RESULT_PUBLISH_FAILED,
        command.task_uuid,
        stage="COMPLETED",
        queue=settings.command_queue,
    )


async def _consume_once(settings: ChunkRebuildRabbitMQSettings) -> None:
    """持续消费切片重建命令队列直到连接中断或任务取消。

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
        chunk_rebuild_log(
            ChunkRebuildStage.CONSUMER_CONNECTED,
            queue=settings.command_queue,
            routing_key=settings.command_routing_key,
        )
        async with queue.iterator() as queue_iter:
            async for incoming in queue_iter:
                await _handle_incoming_message(incoming, settings)


async def run_chunk_rebuild_consumer() -> None:
    """启动切片重建命令消费者，并在异常时自动重连。

    Returns:
        None: 常驻协程，除非任务被取消否则不会主动退出。
    """
    settings = get_chunk_rebuild_settings()
    while True:
        try:
            await _consume_once(settings)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            chunk_rebuild_log(
                ChunkRebuildStage.CONSUMER_RECONNECTING,
                error=str(exc),
                retry_after_seconds=5,
            )
            await asyncio.sleep(5)
