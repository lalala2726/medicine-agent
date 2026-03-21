from __future__ import annotations

from app.agent.client.state import AgentState
from app.core.config_sync import AgentChatModelSlot

# client 侧兼容辅助函数默认返回的聊天槽位。
DEFAULT_MODEL_SLOT = AgentChatModelSlot.CLIENT_CHAT


def model_switch(state: AgentState) -> AgentChatModelSlot:
    """
    返回 client 通用聊天槽位。

    Args:
        state: client 当前状态；保留入参仅用于兼容局部调用签名。

    Returns:
        AgentChatModelSlot: 固定返回 `clientAssistant.chatModel` 对应槽位。
    """

    _ = state
    return DEFAULT_MODEL_SLOT
