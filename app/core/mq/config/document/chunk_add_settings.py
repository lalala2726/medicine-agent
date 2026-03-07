"""手工新增切片链路 MQ 配置。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.core.mq.config.document._shared import DocumentRabbitMQSettingsBase
from app.core.mq.config.topology import (
    MQ_EXCHANGE_KNOWLEDGE_CHUNK_ADD,
    MQ_QUEUE_CHUNK_ADD_COMMAND,
    MQ_ROUTING_KEY_CHUNK_ADD_COMMAND,
    MQ_ROUTING_KEY_CHUNK_ADD_RESULT,
)


@dataclass(frozen=True)
class ChunkAddRabbitMQSettings(DocumentRabbitMQSettingsBase):
    """手工新增切片命令消费与结果发布使用的 MQ 配置。

    Attributes:
        exchange: 命令和结果共用的交换机名称。
        command_queue: AI 服务消费的命令队列名称。
        command_routing_key: 业务侧发布命令使用的 routing key。
        result_routing_key: AI 服务发布结果使用的 routing key。
        prefetch_count: 单消费者预取数量。
    """


@lru_cache(maxsize=1)
def get_chunk_add_settings() -> ChunkAddRabbitMQSettings:
    """返回手工新增切片链路固定约定的 MQ 配置。

    Returns:
        ChunkAddRabbitMQSettings: 约定好的配置对象。
    """
    return ChunkAddRabbitMQSettings(
        exchange=MQ_EXCHANGE_KNOWLEDGE_CHUNK_ADD,
        command_queue=MQ_QUEUE_CHUNK_ADD_COMMAND,
        command_routing_key=MQ_ROUTING_KEY_CHUNK_ADD_COMMAND,
        result_routing_key=MQ_ROUTING_KEY_CHUNK_ADD_RESULT,
    )
