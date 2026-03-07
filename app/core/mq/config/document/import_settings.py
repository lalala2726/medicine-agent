"""知识库导入链路 MQ 配置。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.core.mq.config.document._shared import DocumentRabbitMQSettingsBase
from app.core.mq.config.topology import (
    MQ_EXCHANGE_KNOWLEDGE_IMPORT,
    MQ_QUEUE_IMPORT_COMMAND,
    MQ_ROUTING_KEY_IMPORT_COMMAND,
    MQ_ROUTING_KEY_IMPORT_RESULT,
)

IMPORT_LATEST_VERSION_KEY_PREFIX = "kb:latest"


@dataclass(frozen=True)
class ImportRabbitMQSettings(DocumentRabbitMQSettingsBase):
    """导入命令消费与结果发布使用的 MQ 配置。"""

    latest_version_key_prefix: str = IMPORT_LATEST_VERSION_KEY_PREFIX


@lru_cache(maxsize=1)
def get_import_settings() -> ImportRabbitMQSettings:
    """返回导入链路固定约定的 MQ 配置。

    Returns:
        ImportRabbitMQSettings: 约定好的配置对象。
    """
    return ImportRabbitMQSettings(
        exchange=MQ_EXCHANGE_KNOWLEDGE_IMPORT,
        command_queue=MQ_QUEUE_IMPORT_COMMAND,
        command_routing_key=MQ_ROUTING_KEY_IMPORT_COMMAND,
        result_routing_key=MQ_ROUTING_KEY_IMPORT_RESULT,
    )
