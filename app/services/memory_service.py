from typing import Literal

from core.codes import ResponseCode
from core.exceptions import ServiceException
from schemas.memory import Memory
from services.conversation_service import conversation_exists, get_conversation

# 最大窗口记忆数
_MEMORY_MAX_WINDOW = 100


def load_memory(
        *,
        memory_type: Literal["window", "summary"],
        conversation_id: str,
        user_id: int
) -> Memory:
    """
    加载记忆。

    Args:
        memory_type (Literal["window", "summary"]): 记忆类型，必须是 'window' 或 'summary'
            - 'window': 窗口记忆，用于存储窗口记忆内容。
            - 'summary': 超过指定的会话数会进行总结。
        conversation_id (str): 会话ID。
        user_id (int): 用户ID。

    Returns:
        str: 记忆内容。
    """
    match memory_type:
        case "window":
           return load_memory_by_window(conversation_id=conversation_id, user_id=user_id)
        case "summary":
            return load_memory_by_summary(conversation_id=conversation_id, user_id=user_id)
        case _:
            raise ValueError(f"Invalid memory type: {memory_type}")



def load_memory_by_window(
        conversation_id: str,
        user_id: int
) -> Memory:

    if not conversation_exists(conversation_id):
        raise ServiceException(code=ResponseCode.NOT_FOUND,message="会话不存在")


def load_memory_by_summary(
        conversation_id: str,
        user_id: int
) -> Memory:
    pass