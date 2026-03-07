from __future__ import annotations

from loguru import logger

from app.core.mq._aio_pika_loader import load_aio_pika_publisher
from app.core.mq.config.document.chunk_add_settings import get_chunk_add_settings
from app.core.mq.contracts.document.chunk_add_models import KnowledgeChunkAddResultMessage


async def publish_chunk_add_result(
        message_payload: KnowledgeChunkAddResultMessage,
) -> bool:
    """发布单条手工新增切片结果消息。

    Args:
        message_payload: 待投递的结果消息对象。

    Returns:
        bool: 消息成功投递到目标 exchange 时返回 ``True``，否则返回 ``False``。
    """
    connect_robust, exchange_type_enum, message_cls, delivery_mode_enum = load_aio_pika_publisher()
    settings = get_chunk_add_settings()

    try:
        connection = await connect_robust(settings.url)
        async with connection:
            channel = await connection.channel(publisher_confirms=True)
            exchange = await channel.declare_exchange(
                settings.exchange,
                exchange_type_enum.DIRECT,
                durable=True,
            )
            message = message_cls(
                body=message_payload.to_json_bytes(),
                content_type="application/json",
                delivery_mode=delivery_mode_enum.PERSISTENT,
            )
            await exchange.publish(message, routing_key=settings.result_routing_key)
        logger.info(
            "手工新增切片结果消息投递成功: task_uuid={}, stage={}, routing_key={}",
            message_payload.task_uuid,
            message_payload.stage,
            settings.result_routing_key,
        )
        return True
    except Exception as exc:
        logger.error(
            "手工新增切片结果消息投递失败: task_uuid={}, stage={}, routing_key={}, error={}",
            message_payload.task_uuid,
            message_payload.stage,
            settings.result_routing_key,
            exc,
        )
        return False
