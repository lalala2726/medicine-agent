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
DEFAULT_CONVERSATION_SUMMARIES_COLLECTION = "conversation_summaries"
DEFAULT_MONGODB_STARTUP_PING_ENABLED = False


def _parse_timeout_ms(value: Optional[str]) -> int:
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
    创建并缓存 MongoDB 客户端。

    说明：
    - 统一在这里管理连接超时，避免各业务服务重复配置。
    - 客户端本身线程安全，适合作为进程级单例复用。
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
    db_name = (os.getenv("MONGODB_DB_NAME") or DEFAULT_MONGODB_DB_NAME).strip()
    if not db_name:
        raise ServiceException(
            code=ResponseCode.INTERNAL_ERROR,
            message="MONGODB_DB_NAME 不能为空",
        )
    return get_mongo_client()[db_name]


def verify_mongodb_connection() -> None:
    """
    执行 MongoDB 连通性检查。

    启动期会调用该函数做 fail-fast，避免请求阶段才暴露鉴权/连通性问题。
    """

    try:
        get_mongo_client().admin.command("ping")
    except PyMongoError as exc:
        raise ServiceException(
            code=ResponseCode.INTERNAL_ERROR,
            message=f"MongoDB 连接校验失败: {exc}",
        ) from exc
