from __future__ import annotations

from app.agent.admin.state import AgentState
from app.core.config_sync import AgentChatModelSlot

# 默认业务模型槽位：未识别任务难度时一律使用管理端 simple。
DEFAULT_MODEL_SLOT = AgentChatModelSlot.ADMIN_BUSINESS_SIMPLE

# 按任务难度选择模型槽位：普通(normal) -> 管理端 simple，高(high) -> 管理端 complex。
_DIFFICULTY_MODEL_SLOT_MAP: dict[str, AgentChatModelSlot] = {
    "normal": AgentChatModelSlot.ADMIN_BUSINESS_SIMPLE,
    "high": AgentChatModelSlot.ADMIN_BUSINESS_COMPLEX,
}


def model_switch(state: AgentState) -> AgentChatModelSlot:
    """根据 Gateway 输出的任务难度选择业务模型槽位。

    Args:
        state: LangGraph 节点状态，期望包含 ``routing.task_difficulty``。

    Returns:
        ``normal`` 时返回 ``adminAssistant.businessNodeSimpleModel``；
        ``high`` 时返回 ``adminAssistant.businessNodeComplexModel``；
        缺失或未知值时回退到 ``adminAssistant.businessNodeSimpleModel``。
    """

    routing = state.get("routing")
    task_difficulty = (
        str(routing.get("task_difficulty") or "").strip().lower()
        if isinstance(routing, dict)
        else ""
    )
    return _DIFFICULTY_MODEL_SLOT_MAP.get(task_difficulty, DEFAULT_MODEL_SLOT)
