"""切片重建链路 MQ 配置。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.core.mq.config.document._shared import DocumentRabbitMQSettingsBase
from app.core.mq.config.topology import (
    MQ_EXCHANGE_KNOWLEDGE_CHUNK_REBUILD,
    MQ_QUEUE_CHUNK_REBUILD_COMMAND,
    MQ_ROUTING_KEY_CHUNK_REBUILD_COMMAND,
    MQ_ROUTING_KEY_CHUNK_REBUILD_RESULT,
)

# 单切片编辑场景的 latest-version Redis key 前缀。
CHUNK_EDIT_LATEST_VERSION_KEY_PREFIX = "kb:chunk_edit:latest_version"


@dataclass(frozen=True)
class ChunkRebuildRabbitMQSettings(DocumentRabbitMQSettingsBase):
    """切片重建命令消费与结果发布使用的 MQ 配置。

    Attributes:
        exchange: 命令和结果共用的交换机名称。
        command_queue: AI 服务消费的命令队列名称。
        command_routing_key: 业务侧发布命令使用的 routing key。
        result_routing_key: AI 服务发布结果使用的 routing key。
        prefetch_count: 单消费者预取数量。
        latest_version_key_prefix: 切片编辑 latest-version Redis key 前缀。
    """

    latest_version_key_prefix: str = CHUNK_EDIT_LATEST_VERSION_KEY_PREFIX


@lru_cache(maxsize=1)
def get_chunk_rebuild_settings() -> ChunkRebuildRabbitMQSettings:
    """返回切片重建链路固定约定的 MQ 配置。

    Returns:
        ChunkRebuildRabbitMQSettings: 约定好的配置对象。
    """
    return ChunkRebuildRabbitMQSettings(
        exchange=MQ_EXCHANGE_KNOWLEDGE_CHUNK_REBUILD,
        command_queue=MQ_QUEUE_CHUNK_REBUILD_COMMAND,
        command_routing_key=MQ_ROUTING_KEY_CHUNK_REBUILD_COMMAND,
        result_routing_key=MQ_ROUTING_KEY_CHUNK_REBUILD_RESULT,
    )
