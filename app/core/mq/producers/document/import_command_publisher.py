from __future__ import annotations

from app.core.mq.config.document.import_settings import (
    ImportRabbitMQSettings,
    get_import_settings,
)
from app.core.mq.connection import open_publish_channel
from app.core.mq.contracts.document.import_models import KnowledgeImportCommandMessage


async def publish_import_commands(messages: list[KnowledgeImportCommandMessage]) -> None:
    """批量发布导入命令消息。

    Args:
        messages: 待发布的导入命令消息列表。
    """
    if not messages:
        return

    settings = get_import_settings()
    async with open_publish_channel() as mq:
        exchange = await mq.channel.declare_exchange(
            settings.exchange,
            mq.exchange_type_enum.DIRECT,
            durable=True,
        )
        queue = await mq.channel.declare_queue(settings.command_queue, durable=True)
        await queue.bind(exchange, routing_key=settings.command_routing_key)
        for payload in messages:
            message = mq.message_cls(
                body=payload.to_json_bytes(),
                content_type="application/json",
                delivery_mode=mq.delivery_mode_enum.PERSISTENT,
            )
            await exchange.publish(message, routing_key=settings.command_routing_key)
