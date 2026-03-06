"""知识库导入链路 MQ 配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException
from app.core.mq.config.common import DEFAULT_PREFETCH_COUNT, parse_positive_int
from app.core.mq.config.topology import (
    MQ_EXCHANGE_KNOWLEDGE_IMPORT,
    MQ_QUEUE_IMPORT_COMMAND,
    MQ_ROUTING_KEY_IMPORT_COMMAND,
    MQ_ROUTING_KEY_IMPORT_RESULT,
)

_DEFAULT_LATEST_VERSION_KEY_PREFIX = "kb:latest"


@dataclass(frozen=True)
class ImportRabbitMQSettings:
    """导入命令消费与结果发布使用的 MQ 配置。"""

    url: str
    exchange: str
    command_queue: str
    command_routing_key: str
    result_routing_key: str
    prefetch_count: int
    latest_version_key_prefix: str


@lru_cache(maxsize=1)
def get_import_settings() -> ImportRabbitMQSettings:
    """从环境变量加载并校验导入链路 MQ 配置。

    Returns:
        ImportRabbitMQSettings: 解析后的配置对象。

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
    prefetch_count = parse_positive_int(
        os.getenv("RABBITMQ_PREFETCH_COUNT"),
        default=DEFAULT_PREFETCH_COUNT,
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

    return ImportRabbitMQSettings(
        url=url,
        exchange=exchange,
        command_queue=command_queue,
        command_routing_key=command_routing_key,
        result_routing_key=result_routing_key,
        prefetch_count=prefetch_count,
        latest_version_key_prefix=latest_version_key_prefix,
    )
