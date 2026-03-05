from __future__ import annotations

from redis.exceptions import RedisError

from app.core.codes import ResponseCode
from app.core.database.redis.config import get_redis_connection
from app.core.exception.exceptions import ServiceException
from app.core.mq.config.settings import RabbitMQSettings, get_rabbitmq_settings


def build_latest_version_key(*, biz_key: str, settings: RabbitMQSettings) -> str:
    """构建某个业务键对应的 Redis 最新版本 key。

    Args:
        biz_key: 业务对象唯一键。
        settings: 包含最新版本 key 前缀的配置对象。

    Returns:
        str: Redis key。
    """
    return f"{settings.latest_version_key_prefix}:{biz_key}"


def get_latest_version(*, biz_key: str, settings: RabbitMQSettings | None = None) -> int | None:
    """读取某个业务键当前最新版本。

    Args:
        biz_key: 业务对象唯一键。
        settings: 可选配置对象，未传则自动加载。

    Returns:
        int | None: 读取成功返回版本号，未找到返回 None。

    Raises:
        ServiceException: Redis 读取失败或值不是整数时抛出。
    """
    resolved_settings = settings or get_rabbitmq_settings()
    redis_client = get_redis_connection()
    try:
        raw_value = redis_client.get(
            build_latest_version_key(biz_key=biz_key, settings=resolved_settings)
        )
    except RedisError as exc:
        raise ServiceException(
            code=ResponseCode.OPERATION_FAILED,
            message=f"读取 latest version 失败: biz_key={biz_key}, error={exc}",
        ) from exc
    if raw_value is None:
        return None
    if isinstance(raw_value, bytes):
        raw_text = raw_value.decode("utf-8")
    else:
        raw_text = str(raw_value)

    try:
        parsed = int(raw_text)
    except ValueError as exc:
        raise ServiceException(
            code=ResponseCode.OPERATION_FAILED,
            message=f"latest version 不是整数: biz_key={biz_key}",
        ) from exc
    return parsed


def is_stale_message(*, biz_key: str, version: int, settings: RabbitMQSettings | None = None) -> bool:
    """判断入站消息是否为旧版本消息。

    Args:
        biz_key: 业务对象唯一键。
        version: 入站消息携带的版本号。
        settings: 可选配置对象。

    Returns:
        bool: 若入站版本小于 Redis 最新版本则返回 True。

    Raises:
        ServiceException: 读取或解析最新版本失败时抛出。
    """
    latest_version = get_latest_version(biz_key=biz_key, settings=settings)
    if latest_version is None:
        return False
    return version < latest_version
