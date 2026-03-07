"""切片重建链路 latest-version 状态存储。"""

from __future__ import annotations

from app.core.mq.config.document.chunk_rebuild_settings import (
    CHUNK_EDIT_LATEST_VERSION_KEY_PREFIX,
    ChunkRebuildRabbitMQSettings,
    get_chunk_rebuild_settings,
)
from app.core.mq.state._version_store_support import read_version_from_redis


def _resolve_prefix(settings: ChunkRebuildRabbitMQSettings | None = None) -> str:
    """解析切片编辑 latest-version key 前缀。"""
    if settings is not None:
        return settings.latest_version_key_prefix
    return get_chunk_rebuild_settings().latest_version_key_prefix or CHUNK_EDIT_LATEST_VERSION_KEY_PREFIX


def build_version_key(
        *,
        vector_id: int,
        settings: ChunkRebuildRabbitMQSettings | None = None,
) -> str:
    """构建单个切片编辑 latest-version Redis key。

    Args:
        vector_id: Milvus 向量主键 ID。
        settings: 可选的切片重建 MQ 配置对象。

    Returns:
        str: 形如 ``{prefix}:{vector_id}`` 的 Redis key。
    """
    return f"{_resolve_prefix(settings)}:{vector_id}"


def get_latest_version(
        *,
        vector_id: int,
        settings: ChunkRebuildRabbitMQSettings | None = None,
) -> int | None:
    """读取切片编辑场景下当前最新版本号。

    Args:
        vector_id: Milvus 向量主键 ID。
        settings: 可选配置对象，用于确定 Redis key 前缀。

    Returns:
        int | None: 当前最新版本号；若 key 不存在则返回 ``None``。

    Raises:
        ServiceException: Redis 读取失败或版本值无法解析为整数时抛出。
    """
    key = build_version_key(vector_id=vector_id, settings=settings)
    return read_version_from_redis(key, entity_label=f"vector_id={vector_id}")


def is_stale(
        *,
        vector_id: int,
        version: int,
        settings: ChunkRebuildRabbitMQSettings | None = None,
) -> bool:
    """判断切片编辑消息是否已被更新版本替代。

    Args:
        vector_id: Milvus 向量主键 ID。
        version: 当前消息携带的版本号。
        settings: 可选配置对象，用于确定 Redis key 前缀。

    Returns:
        bool: 若消息版本小于 Redis 中记录的最新版本则返回 ``True``。

    Raises:
        ServiceException: Redis 读取失败或版本值非法时抛出。
    """
    latest_version = get_latest_version(vector_id=vector_id, settings=settings)
    if latest_version is None:
        return False
    return version < latest_version
