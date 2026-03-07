"""文档链路 MQ 配置子包。"""

from app.core.mq.config.document.chunk_add_settings import (
    ChunkAddRabbitMQSettings,
    get_chunk_add_settings,
)
from app.core.mq.config.document.chunk_rebuild_settings import (
    CHUNK_EDIT_LATEST_VERSION_KEY_PREFIX_DEFAULT,
    ChunkRebuildRabbitMQSettings,
    get_chunk_rebuild_settings,
)
from app.core.mq.config.document.import_settings import (
    ImportRabbitMQSettings,
    get_import_settings,
)

__all__ = [
    "ChunkAddRabbitMQSettings",
    "get_chunk_add_settings",
    "ChunkRebuildRabbitMQSettings",
    "CHUNK_EDIT_LATEST_VERSION_KEY_PREFIX_DEFAULT",
    "get_chunk_rebuild_settings",
    "ImportRabbitMQSettings",
    "get_import_settings",
]
