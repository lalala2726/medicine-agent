from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from importlib import util as importlib_util

from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException
from app.core.mq.config.topology import (
    MQ_EXCHANGE_KNOWLEDGE_IMPORT,
    MQ_QUEUE_IMPORT_COMMAND,
    MQ_ROUTING_KEY_IMPORT_COMMAND,
    MQ_ROUTING_KEY_IMPORT_RESULT,
)

_DEFAULT_PREFETCH_COUNT = 1
_DEFAULT_LATEST_VERSION_KEY_PREFIX = "kb:latest"


def _parse_bool(value: str | None, *, default: bool) -> bool:
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


def _parse_positive_int(value: str | None, *, default: int, name: str) -> int:
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
    """检查是否已配置 RabbitMQ 地址。

    Returns:
        bool: 已配置返回 True，否则返回 False。
    """
    return bool((os.getenv("RABBITMQ_URL") or "").strip())


def is_mq_consumer_enabled() -> bool:
    """检查是否启用应用内 MQ 消费者。

    Returns:
        bool: 启用返回 True，否则返回 False。
    """
    return _parse_bool(os.getenv("MQ_CONSUMER_ENABLED"), default=True)


def is_aio_pika_installed() -> bool:
    """检查是否已安装 aio-pika 依赖。

    Returns:
        bool: 已安装返回 True，否则返回 False。
    """
    return importlib_util.find_spec("aio_pika") is not None


@dataclass(frozen=True)
class RabbitMQSettings:
    """导入命令消费与结果发布使用的 MQ 配置。"""

    url: str
    exchange: str
    command_queue: str
    command_routing_key: str
    result_routing_key: str
    prefetch_count: int
    latest_version_key_prefix: str


@lru_cache(maxsize=1)
def get_rabbitmq_settings() -> RabbitMQSettings:
    """从环境变量加载并校验 MQ 配置。

    Returns:
        RabbitMQSettings: 解析后的配置对象。

    Raises:
        ServiceException: 必填配置缺失或值非法时抛出。
    """
    url = (os.getenv("RABBITMQ_URL") or "").strip()
    if not url:
        raise ServiceException(
            code=ResponseCode.INTERNAL_ERROR,
            message="未配置 RABBITMQ_URL，无法启动导入 MQ 消费",
        )

    exchange = (os.getenv("RABBITMQ_EXCHANGE") or MQ_EXCHANGE_KNOWLEDGE_IMPORT).strip()
    command_queue = (os.getenv("RABBITMQ_COMMAND_QUEUE") or MQ_QUEUE_IMPORT_COMMAND).strip()
    command_routing_key = (
            os.getenv("RABBITMQ_COMMAND_ROUTING_KEY") or MQ_ROUTING_KEY_IMPORT_COMMAND
    ).strip()
    result_routing_key = (
            os.getenv("RABBITMQ_RESULT_ROUTING_KEY") or MQ_ROUTING_KEY_IMPORT_RESULT
    ).strip()
    prefetch_count = _parse_positive_int(
        os.getenv("RABBITMQ_PREFETCH_COUNT"),
        default=_DEFAULT_PREFETCH_COUNT,
        name="RABBITMQ_PREFETCH_COUNT",
    )
    latest_version_key_prefix = (
            os.getenv("KNOWLEDGE_LATEST_VERSION_KEY_PREFIX")
            or _DEFAULT_LATEST_VERSION_KEY_PREFIX
    ).strip()
    if not latest_version_key_prefix:
        raise ServiceException(
            code=ResponseCode.INTERNAL_ERROR,
            message="KNOWLEDGE_LATEST_VERSION_KEY_PREFIX 不能为空",
        )

    return RabbitMQSettings(
        url=url,
        exchange=exchange,
        command_queue=command_queue,
        command_routing_key=command_routing_key,
        result_routing_key=result_routing_key,
        prefetch_count=prefetch_count,
        latest_version_key_prefix=latest_version_key_prefix,
    )
