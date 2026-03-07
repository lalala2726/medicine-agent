"""RabbitMQ 连接与 channel 打开工具。"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator

from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException
from app.core.mq._aio_pika_loader import (
    load_aio_pika_consumer,
    load_aio_pika_publisher,
)


@dataclass(frozen=True)
class PublishChannelResources:
    """发布消息所需的 RabbitMQ channel 资源。"""

    channel: Any
    exchange_type_enum: Any
    message_cls: Any
    delivery_mode_enum: Any


@dataclass(frozen=True)
class ConsumeChannelResources:
    """消费消息所需的 RabbitMQ channel 资源。"""

    channel: Any
    exchange_type_enum: Any


def get_rabbitmq_url() -> str:
    """读取并校验 RabbitMQ 连接地址。

    Returns:
        str: 非空的 RabbitMQ URL。

    Raises:
        ServiceException: 未配置 `RABBITMQ_URL` 时抛出。
    """
    url = (os.getenv("RABBITMQ_URL") or "").strip()
    if not url:
        raise ServiceException(
            code=ResponseCode.INTERNAL_ERROR,
            message="未配置 RABBITMQ_URL，无法连接 RabbitMQ",
        )
    return url


@asynccontextmanager
async def open_publish_channel() -> AsyncIterator[PublishChannelResources]:
    """打开用于发布消息的 RabbitMQ channel。

    Yields:
        PublishChannelResources: 包含 channel 与消息构造依赖的资源对象。
    """
    connect_robust, exchange_type_enum, message_cls, delivery_mode_enum = (
        load_aio_pika_publisher()
    )
    connection = await connect_robust(get_rabbitmq_url())
    async with connection:
        channel = await connection.channel(publisher_confirms=True)
        yield PublishChannelResources(
            channel=channel,
            exchange_type_enum=exchange_type_enum,
            message_cls=message_cls,
            delivery_mode_enum=delivery_mode_enum,
        )


@asynccontextmanager
async def open_consume_channel(
        *,
        prefetch_count: int,
) -> AsyncIterator[ConsumeChannelResources]:
    """打开用于消费消息的 RabbitMQ channel 并设置预取数量。

    Args:
        prefetch_count: 单消费者预取数量。

    Yields:
        ConsumeChannelResources: 包含 channel 与 exchange 类型枚举的资源对象。
    """
    connect_robust, exchange_type_enum = load_aio_pika_consumer()
    connection = await connect_robust(get_rabbitmq_url())
    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=prefetch_count)
        yield ConsumeChannelResources(
            channel=channel,
            exchange_type_enum=exchange_type_enum,
        )
