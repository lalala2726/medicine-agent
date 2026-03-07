"""切片重建链路 MQ 配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException
from app.core.mq.config.common import DEFAULT_PREFETCH_COUNT, parse_positive_int
from app.core.mq.config.topology import (
    MQ_EXCHANGE_KNOWLEDGE_CHUNK_REBUILD,
    MQ_QUEUE_CHUNK_REBUILD_COMMAND,
    MQ_ROUTING_KEY_CHUNK_REBUILD_COMMAND,
    MQ_ROUTING_KEY_CHUNK_REBUILD_RESULT,
)

# 单切片编辑场景的 latest-version Redis key 默认前缀。
CHUNK_EDIT_LATEST_VERSION_KEY_PREFIX_DEFAULT = "kb:chunk_edit:latest_version"


@dataclass(frozen=True)
class ChunkRebuildRabbitMQSettings:
    """切片重建命令消费与结果发布使用的 MQ 配置。

    Attributes:
        url: RabbitMQ 连接地址。
        exchange: 命令和结果共用的交换机名称。
        command_queue: AI 服务消费的命令队列名称。
        command_routing_key: 业务侧发布命令使用的 routing key。
        result_routing_key: AI 服务发布结果使用的 routing key。
        prefetch_count: 单消费者预取数量。
        latest_version_key_prefix: 切片编辑 latest-version Redis key 前缀。
    """

    url: str
    exchange: str
    command_queue: str
    command_routing_key: str
    result_routing_key: str
    prefetch_count: int
    latest_version_key_prefix: str


@lru_cache(maxsize=1)
def get_chunk_rebuild_settings() -> ChunkRebuildRabbitMQSettings:
    """从环境变量加载并校验切片重建 MQ 配置。

    Returns:
        ChunkRebuildRabbitMQSettings: 解析后的配置对象。

    Raises:
        ServiceException: 必填配置缺失或值非法时抛出。
    """
    url = (os.getenv("RABBITMQ_URL") or "").strip()
    if not url:
        raise ServiceException(
            code=ResponseCode.INTERNAL_ERROR,
            message="未配置 RABBITMQ_URL，无法启动切片重建 MQ 消费",
        )

    exchange = (
        os.getenv("RABBITMQ_CHUNK_REBUILD_EXCHANGE")
        or MQ_EXCHANGE_KNOWLEDGE_CHUNK_REBUILD
    ).strip()
    command_queue = (
        os.getenv("RABBITMQ_CHUNK_REBUILD_COMMAND_QUEUE")
        or MQ_QUEUE_CHUNK_REBUILD_COMMAND
    ).strip()
    command_routing_key = (
        os.getenv("RABBITMQ_CHUNK_REBUILD_COMMAND_ROUTING_KEY")
        or MQ_ROUTING_KEY_CHUNK_REBUILD_COMMAND
    ).strip()
    result_routing_key = (
        os.getenv("RABBITMQ_CHUNK_REBUILD_RESULT_ROUTING_KEY")
        or MQ_ROUTING_KEY_CHUNK_REBUILD_RESULT
    ).strip()
    prefetch_count = parse_positive_int(
        os.getenv("RABBITMQ_PREFETCH_COUNT"),
        default=DEFAULT_PREFETCH_COUNT,
        name="RABBITMQ_PREFETCH_COUNT",
    )
    latest_version_key_prefix = (
        os.getenv("KNOWLEDGE_CHUNK_EDIT_LATEST_VERSION_KEY_PREFIX")
        or CHUNK_EDIT_LATEST_VERSION_KEY_PREFIX_DEFAULT
    ).strip()
    if not latest_version_key_prefix:
        raise ServiceException(
            code=ResponseCode.INTERNAL_ERROR,
            message="KNOWLEDGE_CHUNK_EDIT_LATEST_VERSION_KEY_PREFIX 不能为空",
        )

    return ChunkRebuildRabbitMQSettings(
        url=url,
        exchange=exchange,
        command_queue=command_queue,
        command_routing_key=command_routing_key,
        result_routing_key=result_routing_key,
        prefetch_count=prefetch_count,
        latest_version_key_prefix=latest_version_key_prefix,
    )
