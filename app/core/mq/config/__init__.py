"""MQ 配置子包。"""

from app.core.mq.config.chunk_rebuild_settings import (
    CHUNK_EDIT_LATEST_VERSION_KEY_PREFIX_DEFAULT,
    ChunkRebuildRabbitMQSettings,
    get_chunk_rebuild_settings,
)
from app.core.mq.config.common import (
    has_rabbitmq_url_configured,
    is_aio_pika_installed,
    is_chunk_rebuild_consumer_enabled,
    is_import_consumer_enabled,
)
from app.core.mq.config.import_settings import (
    ImportRabbitMQSettings,
    get_import_settings,
)
from app.core.mq.config.topology import (
    MQ_EXCHANGE_KNOWLEDGE_CHUNK_REBUILD,
    MQ_EXCHANGE_KNOWLEDGE_IMPORT,
    MQ_QUEUE_CHUNK_REBUILD_COMMAND,
    MQ_QUEUE_IMPORT_COMMAND,
    MQ_ROUTING_KEY_CHUNK_REBUILD_COMMAND,
    MQ_ROUTING_KEY_CHUNK_REBUILD_RESULT,
    MQ_ROUTING_KEY_IMPORT_COMMAND,
    MQ_ROUTING_KEY_IMPORT_RESULT,
)

__all__ = [
    "ChunkRebuildRabbitMQSettings",
    "CHUNK_EDIT_LATEST_VERSION_KEY_PREFIX_DEFAULT",
    "get_chunk_rebuild_settings",
    "ImportRabbitMQSettings",
    "get_import_settings",
    "has_rabbitmq_url_configured",
    "is_aio_pika_installed",
    "is_chunk_rebuild_consumer_enabled",
    "is_import_consumer_enabled",
    "MQ_EXCHANGE_KNOWLEDGE_CHUNK_REBUILD",
    "MQ_EXCHANGE_KNOWLEDGE_IMPORT",
    "MQ_QUEUE_CHUNK_REBUILD_COMMAND",
    "MQ_QUEUE_IMPORT_COMMAND",
    "MQ_ROUTING_KEY_CHUNK_REBUILD_COMMAND",
    "MQ_ROUTING_KEY_CHUNK_REBUILD_RESULT",
    "MQ_ROUTING_KEY_IMPORT_COMMAND",
    "MQ_ROUTING_KEY_IMPORT_RESULT",
]
