from __future__ import annotations

from app.core.mq._aio_pika_loader import load_aio_pika_publisher
from app.core.mq.config.document.import_settings import (
    ImportRabbitMQSettings,
    get_import_settings,
)
from app.core.mq.contracts.document.import_models import KnowledgeImportCommandMessage


async def publish_import_commands(messages: list[KnowledgeImportCommandMessage]) -> None:
    """批量发布导入命令消息。

    Args:
        messages: 待发布的导入命令消息列表。
    """
    if not messages:
        return

    connect_robust, exchange_type_enum, message_cls, delivery_mode_enum = load_aio_pika_publisher()
    settings = get_import_settings()
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
