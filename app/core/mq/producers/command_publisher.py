from __future__ import annotations

from typing import Any

from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException
from app.core.mq.config.settings import get_rabbitmq_settings
from app.core.mq.contracts.models import KnowledgeImportCommandMessage


def _load_aio_pika() -> tuple[Any, Any, Any, Any]:
    """懒加载发布消息所需的 aio-pika 组件。

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
            message="缺少 aio-pika 依赖，无法投递 MQ 消息",
        ) from exc
    return connect_robust, ExchangeType, Message, DeliveryMode


async def publish_import_commands(messages: list[KnowledgeImportCommandMessage]) -> None:
    """批量发布导入命令消息。

    Args:
        messages: 待发布的导入命令消息列表。

    Returns:
        None。
    """
    if not messages:
        return

    connect_robust, exchange_type_enum, message_cls, delivery_mode_enum = _load_aio_pika()
    settings = get_rabbitmq_settings()
    connection = await connect_robust(settings.url)
    async with connection:
        channel = await connection.channel(publisher_confirms=True)
        exchange = await channel.declare_exchange(
            settings.exchange,
            exchange_type_enum.DIRECT,
            durable=True,
        )
        queue = await channel.declare_queue(settings.command_queue, durable=True)
        await queue.bind(exchange, routing_key=settings.command_routing_key)
        for payload in messages:
            message = message_cls(
                body=payload.to_json_bytes(),
                content_type="application/json",
                delivery_mode=delivery_mode_enum.PERSISTENT,
            )
            await exchange.publish(message, routing_key=settings.command_routing_key)
