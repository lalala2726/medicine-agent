import os
from functools import lru_cache
from typing import Optional

from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import PyMongoError

from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException

DEFAULT_MONGODB_URI = "mongodb://localhost:27017"
DEFAULT_MONGODB_DB_NAME = "medicine_ai_agent"
DEFAULT_MONGODB_TIMEOUT_MS = 3000
DEFAULT_CONVERSATIONS_COLLECTION = "conversations"
DEFAULT_MESSAGES_COLLECTION = "messages"
DEFAULT_MESSAGE_TRACES_COLLECTION = "message_traces"
DEFAULT_MESSAGE_TTS_USAGES_COLLECTION = "message_tts_usages"
DEFAULT_CONVERSATION_SUMMARIES_COLLECTION = "conversation_summaries"
DEFAULT_MONGODB_STARTUP_PING_ENABLED = False


def _parse_timeout_ms(value: Optional[str]) -> int:
    """
    功能描述:
        解析并校验 MongoDB 超时配置（毫秒）。

    参数说明:
        value (Optional[str]): 环境变量 `MONGODB_TIMEOUT_MS` 原始值；默认值为 `None`。

    返回值:
        int: 合法超时毫秒数。

    异常说明:
        ServiceException:
            - 配置不是整数时抛出；
            - 配置小于等于 0 时抛出。
    """
    if value is None or value == "":
        return DEFAULT_MONGODB_TIMEOUT_MS
    try:
        timeout_ms = int(value)
    except ValueError as exc:
        raise ServiceException(
            code=ResponseCode.INTERNAL_ERROR,
            message="MONGODB_TIMEOUT_MS 必须是整数",
        ) from exc

    if timeout_ms <= 0:
        raise ServiceException(
            code=ResponseCode.INTERNAL_ERROR,
            message="MONGODB_TIMEOUT_MS 必须大于 0",
        )
    return timeout_ms


@lru_cache(maxsize=1)
def get_mongo_client() -> MongoClient:
    """
    功能描述:
        创建并缓存 MongoDB 客户端，统一管理连接配置。

    参数说明:
        无。

    返回值:
        MongoClient: 进程级复用的 MongoDB 客户端。

    异常说明:
        ServiceException: 当超时配置非法时由 `_parse_timeout_ms` 抛出。
    """
    uri = os.getenv("MONGODB_URI", DEFAULT_MONGODB_URI)
    timeout_ms = _parse_timeout_ms(os.getenv("MONGODB_TIMEOUT_MS"))
    return MongoClient(
        uri,
        serverSelectionTimeoutMS=timeout_ms,
        connectTimeoutMS=timeout_ms,
        socketTimeoutMS=timeout_ms,
    )


def get_mongo_database() -> Database:
    """
    功能描述:
        获取当前服务配置对应的 MongoDB 数据库实例。

    参数说明:
        无。

    返回值:
        Database: 业务 MongoDB 数据库对象。

    异常说明:
        ServiceException: 当 `MONGODB_DB_NAME` 为空字符串时抛出。
    """
    db_name = (os.getenv("MONGODB_DB_NAME") or DEFAULT_MONGODB_DB_NAME).strip()
    if not db_name:
        raise ServiceException(
            code=ResponseCode.INTERNAL_ERROR,
            message="MONGODB_DB_NAME 不能为空",
        )
    return get_mongo_client()[db_name]


def verify_mongodb_connection() -> None:
    """
    功能描述:
        执行 MongoDB 连通性检查，用于启动阶段 fail-fast。

    参数说明:
        无。

    返回值:
        None: 校验成功时无返回值。

    异常说明:
        ServiceException: 当 MongoDB 不可达或鉴权失败时抛出。
    """
    try:
        get_mongo_client().admin.command("ping")
    except PyMongoError as exc:
        raise ServiceException(
            code=ResponseCode.INTERNAL_ERROR,
            message=f"MongoDB 连接校验失败: {exc}",
        ) from exc
