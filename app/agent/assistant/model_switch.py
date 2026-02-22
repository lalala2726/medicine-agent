from __future__ import annotations

from app.agent.assistant.state import AgentState

DEFAULT_MODEL = "qwen-max"

# 按任务难度选择模型：简单 -> flash，普通 -> max，复杂 -> qwen3-plus。
_DIFFICULTY_MODEL_MAP: dict[str, str] = {
    "simple": "qwen-flash",
    "normal": "qwen-max",
    "complex": "qwen3-plus",
}


def model_switch(state: AgentState) -> str:
    """
    根据 gateway 产出的任务难度选择模型。

    规则：
    - simple  -> qwen-flash
    - normal  -> qwen-max
    - complex -> qwen3-plus
    - 无法获取任务难度时，返回普通模型（qwen-max）
    """

    routing = state.get("routing")

    task_difficulty = str(routing.get("task_difficulty") or "").strip().lower()
    if not task_difficulty:
        return DEFAULT_MODEL

    return _DIFFICULTY_MODEL_MAP.get(task_difficulty, DEFAULT_MODEL)
