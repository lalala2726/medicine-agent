"""aio-pika 懒加载工具。

所有 MQ producer / consumer 共用此模块加载 aio-pika 组件，
避免在每个文件中重复实现相同的懒加载逻辑。
"""

from __future__ import annotations

from typing import Any

from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException


def load_aio_pika_publisher() -> tuple[Any, Any, Any, Any]:
    """懒加载发布消息所需的 aio-pika 组件。

    Returns:
        tuple: ``(connect_robust, ExchangeType, Message, DeliveryMode)``。

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


def load_aio_pika_consumer() -> tuple[Any, Any]:
    """懒加载消费者所需的 aio-pika 组件。

    Returns:
        tuple: ``(connect_robust, ExchangeType)``。

    Raises:
        ServiceException: 未安装 aio-pika 时抛出。
    """
    try:
        from aio_pika import ExchangeType, connect_robust
    except Exception as exc:
        raise ServiceException(
            code=ResponseCode.INTERNAL_ERROR,
            message="缺少 aio-pika 依赖，无法启动 MQ 消费者",
        ) from exc
    return connect_robust, ExchangeType
