import datetime
from typing import Annotated

from pydantic import Field
from pymongo.errors import PyMongoError

from core.codes import ResponseCode
from core.exceptions import ServiceException
from core.mongodb import get_mongo_database

_TABLE_NAME = "conversations"
_ADMIN_MARK = "admin"
_USER_MARK = "user"


def _get_conversation(
        *,
        conversation_uuid: str,
        conversation_type: str,
        user_id: int,
) -> dict | None:
    """
    获取会话的内部实现

    Args:
        conversation_uuid: 会话的唯一标识符
        conversation_type: 会话类型 (admin/user)
        user_id: 用户ID，可选，用于验证会话归属

    Returns:
        dict | None: 如果找到匹配的会话则返回会话数据字典，否则返回None

    Raises:
        ServiceException: 当数据库操作出现错误时抛出异常
    """
    db = get_mongo_database()
    collection = db[_TABLE_NAME]

    query = {
        "uuid": conversation_uuid,
        "conversation_type": conversation_type,
        "user_id": user_id,
    }

    try:
        return collection.find_one(query)
    except PyMongoError as exc:
        raise ServiceException(code=ResponseCode.DATABASE_ERROR, message="数据库错误") from exc


def get_admin_conversation(
        *,
        conversation_uuid: Annotated[str, Field(min_length=1)],
        user_id: Annotated[int, Field(ge=1)],
) -> dict | None:
    """
    获取管理端会话

    根据提供的会话UUID和用户ID，从数据库中查询对应的管理端会话记录

    Args:
        conversation_uuid (str): 会话的唯一标识符
        user_id (int): 用户的唯一标识符，用于验证会话归属

    Returns:
        dict | None: 如果找到匹配的会话则返回会话数据字典，否则返回None

    Raises:
        ServiceException: 当数据库操作出现错误时抛出异常
    """
    return _get_conversation(
        conversation_uuid=conversation_uuid,
        conversation_type=_ADMIN_MARK,
        user_id=user_id,
    )


def get_client_conversation(
        *,
        conversation_uuid: Annotated[str, Field(min_length=1)],
        user_id: Annotated[int, Field(ge=1)],
) -> dict | None:
    """
    获取客户端会话

    根据提供的会话UUID，从数据库中查询对应的客户端会话记录

    Args:
        conversation_uuid (str): 会话的唯一标识符

    Returns:
        dict | None: 如果找到匹配的会话则返回会话数据字典，否则返回None

    Raises:
        ServiceException: 当数据库操作出现错误时抛出异常
        :param conversation_uuid: 会话唯一标识UUID
        :param user_id: 用户ID
    """
    return _get_conversation(
        user_id=user_id,
        conversation_uuid=conversation_uuid,
        conversation_type=_USER_MARK,
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
        conversation_type: 会话类型 (admin/user)
        user_id: 用户ID，用于关联会话归属

    Returns:
        str: _id (MongoDB ObjectId 字符串)

    Raises:
        ServiceException: 当数据库操作出现错误时抛出异常
    """
    db = get_mongo_database()
    collection = db[_TABLE_NAME]

    now = datetime.datetime.now()

    conversation = {
        "uuid": conversation_uuid,
        "conversation_type": conversation_type,
        "user_id": user_id,
        "title": "新聊天",
        "create_time": now,
        "update_time": now,
        "message_count": 0,
    }

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
        conversation_type=_USER_MARK,
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
