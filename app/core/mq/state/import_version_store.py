"""知识库导入链路 latest-version 状态存储。"""

from __future__ import annotations

from app.core.mq.config.import_settings import ImportRabbitMQSettings, get_import_settings
from app.core.mq.state._version_store_support import read_version_from_redis


def build_version_key(*, biz_key: str, settings: ImportRabbitMQSettings) -> str:
    """构建某个业务键对应的 Redis 最新版本 key。

    Args:
        biz_key: 业务对象唯一键。
        settings: 包含最新版本 key 前缀的配置对象。

    Returns:
        str: 形如 ``{prefix}:{biz_key}`` 的 Redis key。
    """
    return f"{settings.latest_version_key_prefix}:{biz_key}"


def get_latest_version(*, biz_key: str, settings: ImportRabbitMQSettings | None = None) -> int | None:
    """读取某个业务键当前最新版本。

    Args:
        biz_key: 业务对象唯一键。
        settings: 可选配置对象，未传则自动加载。

    Returns:
        int | None: 读取成功返回版本号，未找到返回 None。

    Raises:
        ServiceException: Redis 读取失败或值不是整数时抛出。
    """
    resolved_settings = settings or get_import_settings()
    key = build_version_key(biz_key=biz_key, settings=resolved_settings)
    return read_version_from_redis(key, entity_label=f"biz_key={biz_key}")


def is_stale(*, biz_key: str, version: int, settings: ImportRabbitMQSettings | None = None) -> bool:
    """判断入站消息是否为旧版本消息。

    Args:
        biz_key: 业务对象唯一键。
        version: 入站消息携带的版本号。
        settings: 可选配置对象。

    Returns:
        bool: 若入站版本小于 Redis 最新版本则返回 ``True``。

    Raises:
        ServiceException: 读取或解析最新版本失败时抛出。
    """
    latest_version = get_latest_version(biz_key=biz_key, settings=settings)
    if latest_version is None:
        return False
    return version < latest_version
