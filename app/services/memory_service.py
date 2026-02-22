from typing import Literal

from langchain_core.messages import HumanMessage, AIMessage

from core.codes import ResponseCode
from core.exceptions import ServiceException
from core.request_context import get_user_id
from schemas.memory import Memory
from services.conversation_service import conversation_exists, get_conversation
from services.message_service import get_message_by_uuid, list_messages

# 最大窗口记忆数
_MEMORY_MAX_WINDOW = 100


def load_memory(
        *,
        memory_type: Literal["window", "summary"],
        conversation_uuid: str,
        user_id: int
) -> Memory:
    """
    加载记忆。

    Args:
        memory_type (Literal["window", "summary"]): 记忆类型，必须是 'window' 或 'summary'
            - 'window': 窗口记忆，用于存储窗口记忆内容。
            - 'summary': 超过指定的会话数会进行总结。
        conversation_uuid (str): 会话ID。
        user_id (int): 用户ID。

    Returns:
        str: 记忆内容。
    """
    match memory_type:
        case "window":
           return load_memory_by_window(conversation_uuid=conversation_uuid, user_id=user_id)
        case "summary":
            return load_memory_by_summary(conversation_uuid=conversation_uuid, user_id=user_id)
        case _:
            raise ValueError(f"Invalid memory type: {memory_type}")



def load_memory_by_window(conversation_uuid: str) -> Memory:

    if not conversation_exists(conversation_uuid):
        raise ServiceException(code=ResponseCode.NOT_FOUND,message="会话不存在")
    user_id:int = get_user_id()
    conversation_id = get_conversation(conversation_uuid=conversation_uuid,user_id=user_id).id
    message_documents = list_messages(
        conversation_id=conversation_id,
        limit=_MEMORY_MAX_WINDOW,
        ascending=False,
    )
    history_messages: list[HumanMessage | AIMessage] = [
        HumanMessage(content=document.content)
        if document.role == "user"
        else AIMessage(content=document.content)
        for document in message_documents
    ]
    _ = history_messages

def load_memory_by_summary(
        conversation_uuid: str,
        user_id: int
) -> Memory:
    pass
