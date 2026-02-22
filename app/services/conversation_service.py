import datetime
import os
from typing import Annotated, Any, Mapping

from bson.int64 import Int64
from pydantic import Field
from pymongo import DESCENDING
from pymongo.errors import PyMongoError

from app.core.codes import ResponseCode
from app.core.exceptions import ServiceException
from app.core.mongodb import DEFAULT_CONVERSATIONS_COLLECTION, get_mongo_database
from app.schemas.document.conversation import (
    ConversationCreate,
    ConversationDocument,
    ConversationListItem,
    ConversationType,
    ConversationUpdateSet,
)

_TABLE_NAME: str = (
        (os.getenv("MONGODB_CONVERSATIONS_COLLECTION") or DEFAULT_CONVERSATIONS_COLLECTION).strip()
        or DEFAULT_CONVERSATIONS_COLLECTION
)
"""数据库中会话集合的名称"""

_ADMIN_MARK: str = "admin"
"""管理端会话类型的标识符"""

_CLIENT_MARK: str = "client"
"""客户端会话类型的标识符"""

_NOT_DELETED_FILTER: dict[str, int] = {"$ne": 1}
"""用于查询未删除记录的MongoDB过滤条件"""


def _to_mongo_long(value: int) -> Int64:
    """
    将 Python int 显式转换为 MongoDB int64。

    这样可以避免在开启 `$jsonSchema`（bsonType=long）校验时，
    Python 小整数被编码成 int32 导致写入失败。
    """

    return Int64(value)


def _to_conversation_document(document: Mapping[str, Any]) -> ConversationDocument:
    """把 Mongo 原始文档转换为会话文档模型。"""

    return ConversationDocument.model_validate(document)


def get_conversation(
        *,
        conversation_uuid: str,
        conversation_type: str | None = None,
        user_id: int,
) -> ConversationDocument | None:
    """
    获取会话的内部实现

    Args:
        conversation_uuid: 会话的唯一标识符
        conversation_type: 会话类型 (admin/client),可选
        user_id: 用户ID，可选，用于验证会话归属

    Returns:
        ConversationDocument | None: 如果找到匹配的会话则返回会话模型，否则返回 None

    Raises:
        ServiceException: 当数据库操作出现错误时抛出异常
    """
    db = get_mongo_database()
    collection = db[_TABLE_NAME]

    resolved_conversation_type: str | dict[str, list[str]]
    if conversation_type is None:
        resolved_conversation_type = {"$in": [_ADMIN_MARK, _CLIENT_MARK]}
    else:
        resolved_conversation_type = conversation_type

    query = {
        "uuid": conversation_uuid,
        "conversation_type": resolved_conversation_type,
        "user_id": _to_mongo_long(user_id),
        "is_deleted": _NOT_DELETED_FILTER,
    }

    try:
        document = collection.find_one(query)
    except PyMongoError as exc:
        raise ServiceException(code=ResponseCode.DATABASE_ERROR, message="数据库错误") from exc

    if document is None:
        return None
    return _to_conversation_document(document)


def get_admin_conversation(
        *,
        conversation_uuid: Annotated[str, Field(min_length=1)],
        user_id: Annotated[int, Field(ge=1)],
) -> ConversationDocument | None:
    """
    获取管理端会话

    根据提供的会话UUID和用户ID，从数据库中查询对应的管理端会话记录

    Args:
        conversation_uuid (str): 会话的唯一标识符
        user_id (int): 用户的唯一标识符，用于验证会话归属

    Returns:
        ConversationDocument | None: 如果找到匹配的会话则返回会话模型，否则返回 None

    Raises:
        ServiceException: 当数据库操作出现错误时抛出异常
    """
    return get_conversation(
        conversation_uuid=conversation_uuid,
        conversation_type=_ADMIN_MARK,
        user_id=user_id,
    )


def get_client_conversation(
        *,
        conversation_uuid: Annotated[str, Field(min_length=1)],
        user_id: Annotated[int, Field(ge=1)],
) -> ConversationDocument | None:
    """
    获取客户端会话

    根据提供的会话UUID，从数据库中查询对应的客户端会话记录

    Args:
        conversation_uuid (str): 会话的唯一标识符

    Returns:
        ConversationDocument | None: 如果找到匹配的会话则返回会话模型，否则返回 None

    Raises:
        ServiceException: 当数据库操作出现错误时抛出异常
        :param conversation_uuid: 会话唯一标识UUID
        :param user_id: 用户ID
    """
    return get_conversation(
        user_id=user_id,
        conversation_uuid=conversation_uuid,
        conversation_type=_CLIENT_MARK,
    )


def _add_conversation(
        *,
        conversation_uuid: Annotated[str, Field(min_length=1)],
        conversation_type: Annotated[str, Field(min_length=1)],
        user_id: Annotated[int, Field(ge=1)],
) -> str:
    """
    新增会话的内部实现

    Args:
        conversation_uuid: 会话的唯一标识符
        conversation_type: 会话类型 (admin/client)
        user_id: 用户ID，用于关联会话归属

    Returns:
        str: _id (MongoDB ObjectId 字符串)

    Raises:
        ServiceException: 当数据库操作出现错误时抛出异常
    """
    db = get_mongo_database()
    collection = db[_TABLE_NAME]

    payload = ConversationCreate(
        uuid=conversation_uuid,
        conversation_type=ConversationType(conversation_type),
        user_id=user_id,
    )
    # 统一从模型序列化到 Mongo 文档，避免散落的手写字段。
    conversation = payload.model_dump(mode="python")

    try:
        result = collection.insert_one(conversation)
        return str(result.inserted_id)
    except PyMongoError as exc:
        raise ServiceException(code=ResponseCode.DATABASE_ERROR, message="数据库错误") from exc


def add_client_conversation(
        *,
        conversation_uuid: Annotated[str, Field(min_length=1)],
        user_id: Annotated[int, Field(ge=1)],
) -> str:
    """
    新增客户端会话

    根据提供的会话UUID和用户ID，在数据库中创建一条新的客户端会话记录

    Args:
        conversation_uuid: 会话的唯一标识符
        user_id: 用户的唯一标识符

    Returns:
        str: _id (MongoDB ObjectId 字符串)

    Raises:
        ServiceException: 当数据库操作出现错误时抛出异常
    """
    return _add_conversation(
        conversation_uuid=conversation_uuid,
        conversation_type=_CLIENT_MARK,
        user_id=user_id,
    )


def add_admin_conversation(
        *,
        conversation_uuid: Annotated[str, Field(min_length=1)],
        user_id: Annotated[int, Field(ge=1)],
) -> str:
    """
    新增管理端会话

    根据提供的会话UUID和用户ID，在数据库中创建一条新的管理端会话记录

    Args:
        conversation_uuid: 会话的唯一标识符
        user_id: 用户的唯一标识符

    Returns:
        str: _id (MongoDB ObjectId 字符串)

    Raises:
        ServiceException: 当数据库操作出现错误时抛出异常
    """
    return _add_conversation(
        conversation_uuid=conversation_uuid,
        conversation_type=_ADMIN_MARK,
        user_id=user_id,
    )


def list_admin_conversations(
        *,
        user_id: Annotated[int, Field(ge=1)],
        page_num: Annotated[int, Field(ge=1)] = 1,
        page_size: Annotated[int, Field(ge=1, le=100)] = 20,
) -> tuple[list[ConversationListItem], int]:
    """
    分页查询管理端会话列表（仅返回会话 UUID 与标题）。

    Args:
        user_id: 用户 ID，用于筛选当前用户的会话。
        page_num: 页码（从 1 开始）。
        page_size: 每页大小（1-100）。

    Returns:
        tuple[list[ConversationListItem], int]:
            - rows: 会话列表项模型列表。
            - total: 总记录数。

    Raises:
        ServiceException: 当数据库操作失败时抛出异常。
    """

    db = get_mongo_database()
    collection = db[_TABLE_NAME]

    query = {
        "conversation_type": _ADMIN_MARK,
        "user_id": _to_mongo_long(user_id),
        "is_deleted": _NOT_DELETED_FILTER,
    }
    projection = {
        "_id": 0,
        "uuid": 1,
        "title": 1,
    }
    skip = (page_num - 1) * page_size

    try:
        total = collection.count_documents(query)
        cursor = (
            collection.find(query, projection)
            .sort("update_time", DESCENDING)
            .skip(skip)
            .limit(page_size)
        )
    except PyMongoError as exc:
        raise ServiceException(code=ResponseCode.DATABASE_ERROR, message="数据库错误") from exc

    rows: list[ConversationListItem] = []
    for item in cursor:
        conversation_uuid = str(item.get("uuid") or "").strip()
        if not conversation_uuid:
            continue
        title = str(item.get("title") or "").strip() or "新聊天"
        rows.append(
            ConversationListItem(
                conversation_uuid=conversation_uuid,
                title=title,
            )
        )
    return rows, total


def save_conversation_title(
        *,
        conversation_uuid: Annotated[str, Field(min_length=1)],
        title: Annotated[str, Field(min_length=1)],
) -> None:
    """
    保存会话标题并刷新更新时间。

    Args:
        conversation_uuid: 会话唯一标识 UUID。
        title: 新标题。

    Raises:
        ServiceException: 当数据库操作失败时抛出。
    """

    db = get_mongo_database()
    collection = db[_TABLE_NAME]

    now = datetime.datetime.now()
    query = {
        "uuid": conversation_uuid,
        "is_deleted": _NOT_DELETED_FILTER,
    }
    update_set = ConversationUpdateSet(title=title, update_time=now)
    update_doc = {"$set": update_set.model_dump(mode="python", exclude_none=True)}

    try:
        collection.update_one(query, update_doc)
    except PyMongoError as exc:
        raise ServiceException(code=ResponseCode.DATABASE_ERROR, message="数据库错误") from exc


def update_admin_conversation_title(
        *,
        conversation_uuid: Annotated[str, Field(min_length=1)],
        user_id: Annotated[int, Field(ge=1)],
        title: Annotated[str, Field(min_length=1)],
) -> bool:
    """
    更新当前用户的管理端会话标题。

    Args:
        conversation_uuid: 会话 UUID。
        user_id: 当前用户 ID。
        title: 新标题。

    Returns:
        bool: True 表示命中并更新成功；False 表示会话不存在或无权限。

    Raises:
        ServiceException: 数据库异常时抛出。
    """

    db = get_mongo_database()
    collection = db[_TABLE_NAME]

    now = datetime.datetime.now()
    query = {
        "uuid": conversation_uuid,
        "conversation_type": _ADMIN_MARK,
        "user_id": _to_mongo_long(user_id),
        "is_deleted": _NOT_DELETED_FILTER,
    }
    update_set = ConversationUpdateSet(title=title, update_time=now)
    update_doc = {"$set": update_set.model_dump(mode="python", exclude_none=True)}

    try:
        result = collection.update_one(query, update_doc)
        return int(getattr(result, "matched_count", 0)) > 0
    except PyMongoError as exc:
        raise ServiceException(code=ResponseCode.DATABASE_ERROR, message="数据库错误") from exc


def delete_admin_conversation(
        *,
        conversation_uuid: Annotated[str, Field(min_length=1)],
        user_id: Annotated[int, Field(ge=1)],
) -> bool:
    """
    逻辑删除当前用户的管理端会话。

    Args:
        conversation_uuid: 会话 UUID。
        user_id: 当前用户 ID。

    Returns:
        bool: True 表示删除成功；False 表示会话不存在、无权限或已删除。

    Raises:
        ServiceException: 数据库异常时抛出。
    """

    db = get_mongo_database()
    collection = db[_TABLE_NAME]

    now = datetime.datetime.now()
    query = {
        "uuid": conversation_uuid,
        "conversation_type": _ADMIN_MARK,
        "user_id": _to_mongo_long(user_id),
        "is_deleted": _NOT_DELETED_FILTER,
    }
    update_set = ConversationUpdateSet(is_deleted=1, update_time=now)
    update_doc = {"$set": update_set.model_dump(mode="python", exclude_none=True)}

    try:
        result = collection.update_one(query, update_doc)
        return int(getattr(result, "matched_count", 0)) > 0
    except PyMongoError as exc:
        raise ServiceException(code=ResponseCode.DATABASE_ERROR, message="数据库错误") from exc


def conversation_exists(
        conversation_uuid: Annotated[str, Field(min_length=1)]
) -> bool:
    """
    检查会话是否存在

    根据会话UUID检查数据库中是否存在对应的未删除会话记录

    Args:
        conversation_uuid: 会话的唯一标识符

    Returns:
        bool: True 表示会话存在，False 表示会话不存在

    Raises:
        ServiceException: 当数据库操作出现错误时抛出异常
    """
    db = get_mongo_database()
    collection = db[_TABLE_NAME]

    query = {
        "uuid": conversation_uuid,
        "is_deleted": _NOT_DELETED_FILTER,
    }

    count = collection.count_documents(query)
    return count > 0
