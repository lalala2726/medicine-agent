from typing import Literal


def load_memory(
        *,
        type: Literal["client", "admin"],
        conversation_id: str,
        user_id: int
):
    """
    加载记忆。

    Args:
        type (Literal["client", "admin"]): 会话类型，必须是 'client' 或 'admin'
        conversation_id (str): 会话ID。
        user_id (int): 用户ID。

    Returns:
        str: 记忆内容。
    """
    pass