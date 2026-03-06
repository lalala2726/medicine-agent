"""Redis latest-version 读取公共支撑函数。

为 import_version_store / chunk_rebuild_version_store 提供统一的
Redis GET → bytes 解码 → int 解析 → 异常包装 逻辑。
"""

from __future__ import annotations

from redis.exceptions import RedisError

from app.core.codes import ResponseCode
from app.core.database.redis.config import get_redis_connection
from app.core.exception.exceptions import ServiceException


def read_version_from_redis(key: str, *, entity_label: str) -> int | None:
    """从 Redis 读取并解析整数版本号。

    Args:
        key: Redis key。
        entity_label: 用于异常消息中标识实体的描述，如 ``biz_key=demo:1``。

    Returns:
        int | None: 版本号；key 不存在时返回 ``None``。

    Raises:
        ServiceException: Redis 读取失败或值不是整数时抛出。
    """
    redis_client = get_redis_connection()
    try:
        raw_value = redis_client.get(key)
    except RedisError as exc:
        raise ServiceException(
            code=ResponseCode.OPERATION_FAILED,
            message=f"读取 latest version 失败: {entity_label}, error={exc}",
        ) from exc

    if raw_value is None:
        return None

    raw_text = raw_value.decode("utf-8") if isinstance(raw_value, bytes) else str(raw_value)

    try:
        return int(raw_text)
    except ValueError as exc:
        raise ServiceException(
            code=ResponseCode.OPERATION_FAILED,
            message=f"latest version 不是整数: {entity_label}",
        ) from exc
