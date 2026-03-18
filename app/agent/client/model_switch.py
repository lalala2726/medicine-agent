from __future__ import annotations

from app.agent.client.state import AgentState
from app.core.config_sync import AgentChatModelSlot

DEFAULT_MODEL_SLOT = AgentChatModelSlot.BUSINESS_SIMPLE

_DIFFICULTY_MODEL_SLOT_MAP: dict[str, AgentChatModelSlot] = {
    "normal": AgentChatModelSlot.BUSINESS_SIMPLE,
    "high": AgentChatModelSlot.BUSINESS_COMPLEX,
}


def model_switch(state: AgentState) -> AgentChatModelSlot:
    """根据 client gateway 输出的任务难度选择业务模型槽位。"""

    routing = state.get("routing")
    task_difficulty = (
        str(routing.get("task_difficulty") or "").strip().lower()
        if isinstance(routing, dict)
        else ""
    )
    return _DIFFICULTY_MODEL_SLOT_MAP.get(task_difficulty, DEFAULT_MODEL_SLOT)
