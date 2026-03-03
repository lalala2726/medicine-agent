from __future__ import annotations

from typing import Any

from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException
from app.core.mq.models import KnowledgeImportMessage
from app.core.mq.settings import get_rabbitmq_settings


def _load_aio_pika() -> tuple[Any, Any, Any, Any]:
    """
    功能描述:
        懒加载 aio-pika 依赖，避免在模块导入阶段产生硬依赖。

    参数说明:
        无。

    返回值:
        tuple[Any, Any, Any, Any]:
            - connect_robust 函数
            - ExchangeType 枚举
            - Message 类
            - DeliveryMode 枚举

    异常说明:
        ServiceException: 未安装 aio-pika 依赖时抛出。
    """
    try:
        from aio_pika import DeliveryMode, ExchangeType, Message, connect_robust
    except Exception as exc:
        raise ServiceException(
            code=ResponseCode.INTERNAL_ERROR,
            message="缺少 aio-pika 依赖，无法投递 MQ 消息",
        ) from exc
    return connect_robust, ExchangeType, Message, DeliveryMode


async def publish_import_messages(messages: list[KnowledgeImportMessage]) -> None:
    """
    功能描述:
        批量发布知识库导入任务消息到 RabbitMQ。

    参数说明:
        messages (list[KnowledgeImportMessage]): 待发布的消息列表。

    返回值:
        None: 发布完成无返回值。

    异常说明:
        ServiceException: 当 RabbitMQ 配置缺失时由 settings 模块抛出。
        Exception: 连接 RabbitMQ 或发布消息失败时由底层库抛出。
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
        queue = await channel.declare_queue(settings.queue, durable=True)
        await queue.bind(exchange, routing_key=settings.routing_key)
        for payload in messages:
            message = message_cls(
                body=payload.to_json_bytes(),
                content_type="application/json",
                delivery_mode=delivery_mode_enum.PERSISTENT,
            )
            await exchange.publish(message, routing_key=settings.routing_key)
