"""MQ 配置公共工具函数与运行时开关检测。"""

from __future__ import annotations

import os
from importlib import util as importlib_util

from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException


def parse_bool(value: str | None, *, default: bool) -> bool:
    """解析布尔环境变量。

    Args:
        value: 原始环境变量值。
        default: 未配置时的默认值。

    Returns:
        bool: 解析后的布尔值。
    """
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_positive_int(value: str | None, *, default: int, name: str) -> int:
    """解析正整数环境变量。

    Args:
        value: 原始环境变量值。
        default: 未配置时的默认值。
        name: 环境变量名称，用于错误提示。

    Returns:
        int: 解析后的正整数。

    Raises:
        ServiceException: 配置值不是正整数时抛出。
    """
    if value is None or value.strip() == "":
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ServiceException(
            code=ResponseCode.INTERNAL_ERROR,
            message=f"{name} 必须是正整数",
        ) from exc
    if parsed <= 0:
        raise ServiceException(
            code=ResponseCode.INTERNAL_ERROR,
            message=f"{name} 必须大于 0",
        )
    return parsed


def has_rabbitmq_url_configured() -> bool:
    """检查是否已配置 RabbitMQ 地址。"""
    return bool((os.getenv("RABBITMQ_URL") or "").strip())


def is_import_consumer_enabled() -> bool:
    """检查是否启用应用内导入 MQ 消费者。"""
    return parse_bool(os.getenv("MQ_CONSUMER_ENABLED"), default=True)


def is_chunk_rebuild_consumer_enabled() -> bool:
    """检查是否启用应用内切片重建 MQ 消费者。"""
    return parse_bool(os.getenv("MQ_CHUNK_REBUILD_CONSUMER_ENABLED"), default=True)


def is_chunk_add_consumer_enabled() -> bool:
    """检查是否启用应用内手工新增切片 MQ 消费者。"""
    return parse_bool(os.getenv("MQ_CHUNK_ADD_CONSUMER_ENABLED"), default=True)


def is_aio_pika_installed() -> bool:
    """检查是否已安装 aio-pika 依赖。"""
    return importlib_util.find_spec("aio_pika") is not None


# 默认消费者预取数量。
DEFAULT_PREFETCH_COUNT = 1
