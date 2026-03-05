from __future__ import annotations

from typing import Any

from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException
from app.core.mq.config.settings import get_rabbitmq_settings
from app.core.mq.contracts.models import KnowledgeImportResultMessage


def _load_aio_pika() -> tuple[Any, Any, Any, Any]:
    """懒加载发布结果消息所需的 aio-pika 组件。

    Returns:
        tuple[Any, Any, Any, Any]: connect_robust、ExchangeType、Message、DeliveryMode。

    Raises:
        ServiceException: 未安装 aio-pika 时抛出。
    """
    try:
        from aio_pika import DeliveryMode, ExchangeType, Message, connect_robust
    except Exception as exc:
        raise ServiceException(
            code=ResponseCode.INTERNAL_ERROR,
            message="缺少 aio-pika 依赖，无法发布 MQ 结果消息",
        ) from exc
    return connect_robust, ExchangeType, Message, DeliveryMode


async def publish_import_result_message(message_payload: KnowledgeImportResultMessage) -> bool:
    """发布单条导入结果消息。

    Args:
        message_payload: 导入结果事件消息体。

    Returns:
        bool: 发布成功返回 True，失败返回 False。
    """
    connect_robust, exchange_type_enum, message_cls, delivery_mode_enum = _load_aio_pika()
    settings = get_rabbitmq_settings()

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
        return True
    except Exception:
        return False
