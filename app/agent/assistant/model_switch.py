from __future__ import annotations

from app.agent.assistant.state import AgentState

DEFAULT_MODEL = "doubao-seed-2-0-lite-260215"
DEFAULT_THINK = False

# 按任务难度选择模型：普通(normal) -> max，高(high) -> qwen3-plus。
_DIFFICULTY_MODEL_MAP: dict[str, str] = {
    "normal": "doubao-seed-2-0-lite-260215",
    "high": "doubao-seed-2-0-pro-260215",
}

# 按任务难度决定是否开启深度思考：仅 high 开启。
_DIFFICULTY_THINK_MAP: dict[str, bool] = {
    "normal": False,
    "high": True,
}


def model_switch(state: AgentState) -> tuple[str, bool]:
    """
    功能描述：
        根据 gateway 产出的任务难度，返回应使用的模型名称与 think 开关。

    参数说明：
        state (AgentState): LangGraph 节点状态，期望包含 `routing.task_difficulty`。

    返回值：
        tuple[str, bool]:
            - 第一个值为模型名称：
              - normal  -> qwen-max
              - high    -> qwen3-plus
              - 缺省/未知 -> qwen-max
            - 第二个值为 think 开关：
              - high -> True
              - 其他    -> False

    异常说明：
        无显式异常；当输入缺失或异常时回退到默认模型与默认 think。
    """

    routing = state.get("routing")
    task_difficulty = (
        str(routing.get("task_difficulty") or "").strip().lower()
        if isinstance(routing, dict)
        else ""
    )
    model_name = _DIFFICULTY_MODEL_MAP.get(task_difficulty, DEFAULT_MODEL)
    enable_think = _DIFFICULTY_THINK_MAP.get(task_difficulty, DEFAULT_THINK)
    return model_name, enable_think
