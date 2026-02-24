from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage

from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException
from app.schemas.document.message import MessageRole
from app.schemas.memory import Memory
from app.services.conversation_service import get_conversation
from app.services.message_service import list_messages

# 最大窗口记忆数
_MEMORY_MAX_WINDOW = 100


def load_memory(
        *,
        memory_type: Literal["window", "summary"],
        conversation_uuid: str,
        user_id: int,
        limit: int = _MEMORY_MAX_WINDOW,
) -> Memory:
    """
    加载记忆。

    Args:
        memory_type (Literal["window", "summary"]): 记忆类型，必须是 'window' 或 'summary'
            - 'window': 窗口记忆，用于存储窗口记忆内容。
            - 'summary': 超过指定的会话数会进行总结。
        conversation_uuid (str): 会话ID。
        user_id (int): 用户ID。
        limit (int): 窗口大小。

    Returns:
        Memory: 记忆内容。
    """
    match memory_type:
        case "window":
            return load_memory_by_window(
                conversation_uuid=conversation_uuid,
                user_id=user_id,
                limit=limit,
            )
        case "summary":
            return load_memory_by_summary(conversation_uuid=conversation_uuid, user_id=user_id)
        case _:
            raise ValueError(f"Invalid memory type: {memory_type}")


def load_memory_by_window(
        *,
        conversation_uuid: str,
        user_id: int,
        limit: int = _MEMORY_MAX_WINDOW,
) -> Memory:
    """
    加载窗口记忆（按时间正序返回）。

    Returns:
        Memory: 仅包含 HumanMessage/AIMessage，消息顺序为旧 -> 新。
    """

    normalized_uuid = conversation_uuid.strip()
    if not normalized_uuid:
        raise ServiceException(code=ResponseCode.BAD_REQUEST, message="会话UUID不能为空")
    if user_id <= 0:
        raise ServiceException(code=ResponseCode.BAD_REQUEST, message="用户ID不合法")
    if limit <= 0:
        raise ServiceException(code=ResponseCode.BAD_REQUEST, message="窗口大小不合法")

    conversation = get_conversation(
        conversation_uuid=normalized_uuid,
        user_id=user_id,
    )
    if conversation is None:
        raise ServiceException(code=ResponseCode.NOT_FOUND, message="会话不存在")

    conversation_id = conversation.id
    if conversation_id is None:
        raise ServiceException(code=ResponseCode.DATABASE_ERROR, message="会话数据异常")

    message_documents = list_messages(
        conversation_id=conversation_id,
        limit=limit,
        ascending=False,
    )

    history_messages: list[HumanMessage | AIMessage] = [
        HumanMessage(content=document.content)
        if document.role == MessageRole.USER
        else AIMessage(content=document.content)
        for document in reversed(message_documents)
    ]
    return Memory(messages=history_messages)


def load_memory_by_summary(
        conversation_uuid: str,
        user_id: int,
) -> Memory:
    _ = conversation_uuid
    _ = user_id
    raise NotImplementedError("load_memory_by_summary is not implemented")
