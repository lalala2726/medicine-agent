"""手工新增切片链路 MQ 配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException
from app.core.mq.config.common import DEFAULT_PREFETCH_COUNT, parse_positive_int
from app.core.mq.config.topology import (
    MQ_EXCHANGE_KNOWLEDGE_CHUNK_ADD,
    MQ_QUEUE_CHUNK_ADD_COMMAND,
    MQ_ROUTING_KEY_CHUNK_ADD_COMMAND,
    MQ_ROUTING_KEY_CHUNK_ADD_RESULT,
)


@dataclass(frozen=True)
class ChunkAddRabbitMQSettings:
    """手工新增切片命令消费与结果发布使用的 MQ 配置。

    Attributes:
        url: RabbitMQ 连接地址。
        exchange: 命令和结果共用的交换机名称。
        command_queue: AI 服务消费的命令队列名称。
        command_routing_key: 业务侧发布命令使用的 routing key。
        result_routing_key: AI 服务发布结果使用的 routing key。
        prefetch_count: 单消费者预取数量。
    """

    url: str
    exchange: str
    command_queue: str
    command_routing_key: str
    result_routing_key: str
    prefetch_count: int


@lru_cache(maxsize=1)
def get_chunk_add_settings() -> ChunkAddRabbitMQSettings:
    """从环境变量加载并校验手工新增切片 MQ 配置。

    Returns:
        ChunkAddRabbitMQSettings: 解析后的配置对象。

    Raises:
        ServiceException: 必填配置缺失或值非法时抛出。
    """
    url = (os.getenv("RABBITMQ_URL") or "").strip()
    if not url:
        raise ServiceException(
            code=ResponseCode.INTERNAL_ERROR,
            message="未配置 RABBITMQ_URL，无法启动手工新增切片 MQ 消费",
        )

    exchange = (
        os.getenv("RABBITMQ_CHUNK_ADD_EXCHANGE")
        or MQ_EXCHANGE_KNOWLEDGE_CHUNK_ADD
    ).strip()
    command_queue = (
        os.getenv("RABBITMQ_CHUNK_ADD_COMMAND_QUEUE")
        or MQ_QUEUE_CHUNK_ADD_COMMAND
    ).strip()
    command_routing_key = (
        os.getenv("RABBITMQ_CHUNK_ADD_COMMAND_ROUTING_KEY")
        or MQ_ROUTING_KEY_CHUNK_ADD_COMMAND
    ).strip()
    result_routing_key = (
        os.getenv("RABBITMQ_CHUNK_ADD_RESULT_ROUTING_KEY")
        or MQ_ROUTING_KEY_CHUNK_ADD_RESULT
    ).strip()
    prefetch_count = parse_positive_int(
        os.getenv("RABBITMQ_PREFETCH_COUNT"),
        default=DEFAULT_PREFETCH_COUNT,
        name="RABBITMQ_PREFETCH_COUNT",
    )

    return ChunkAddRabbitMQSettings(
        url=url,
        exchange=exchange,
        command_queue=command_queue,
        command_routing_key=command_routing_key,
        result_routing_key=result_routing_key,
        prefetch_count=prefetch_count,
    )
