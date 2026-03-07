from __future__ import annotations

from loguru import logger

from app.core.mq._aio_pika_loader import load_aio_pika_publisher
from app.core.mq.config.document.import_settings import get_import_settings
from app.core.mq.contracts.document.import_models import KnowledgeImportResultMessage


async def publish_import_result(message_payload: KnowledgeImportResultMessage) -> bool:
    """发布单条导入结果消息。

    Args:
        message_payload: 导入结果事件消息体。

    Returns:
        bool: 发布成功返回 True，失败返回 False。
    """
    connect_robust, exchange_type_enum, message_cls, delivery_mode_enum = load_aio_pika_publisher()
    settings = get_import_settings()

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
            "导入结果消息投递成功: task_uuid={}, stage={}, routing_key={}",
            message_payload.task_uuid,
            message_payload.stage,
            settings.result_routing_key,
        )
        return True
    except Exception as exc:
        logger.error(
            "导入结果消息投递失败: task_uuid={}, stage={}, routing_key={}, error={}",
            message_payload.task_uuid,
            message_payload.stage,
            settings.result_routing_key,
            exc,
        )
        return False
