from __future__ import annotations

import json
from typing import Any

from app.agent.admin.agent_state import AgentState, PlanStep, StepOutput


def _is_coordinator_mode(state: AgentState | dict[str, Any]) -> bool:
    # 只有 coordinator 编排路径，才启用“按步骤配置注入上下文”的严格模式。
    # 直连节点（gateway 直达）仍保留历史行为，避免影响已有能力。
    routing = state.get("routing") or {}
    return routing.get("route_target") == "coordinator_agent"


def get_current_step(
        state: AgentState | dict[str, Any],
        node_name: str,
) -> dict | dict[Any, Any]:
    # planner 每一轮会把“本轮要执行的节点步骤定义”放进 current_step_map。
    # 节点通过 node_name 拿到自己的 step 配置（step_id/read_from/final_output...）。
    routing = state.get("routing") or {}
    current_step_map = routing.get("current_step_map") or {}
    if not isinstance(current_step_map, dict):
        return {}
    step = current_step_map.get(node_name)
    return step if isinstance(step, dict) else {}


def build_step_runtime(
        state: AgentState | dict[str, Any],
        node_name: str,
        *,
        default_task_description: str,
) -> dict[str, Any]:
    """
    构建当前节点执行所需的运行时上下文。
    """
    step = get_current_step(state, node_name)
    coordinator_mode = _is_coordinator_mode(state)

    raw_step_id = step.get("step_id")
    step_id = str(raw_step_id).strip() if isinstance(raw_step_id, str) else ""

    task_description = str(step.get("task_description") or "").strip()
    if not task_description:
        task_description = default_task_description

    # 设计约束：
    # - coordinator 模式下默认不读 user/history，必须通过步骤开关显式开启；
    # - 非 coordinator 模式沿用旧行为（user_input 默认可读）。
    include_user_input = bool(step.get("include_user_input")) if coordinator_mode else True
    include_chat_history = bool(step.get("include_chat_history")) if coordinator_mode else False

    read_from: list[str] = []
    if coordinator_mode and isinstance(step.get("read_from"), list):
        # read_from 只允许在 coordinator 模式生效，避免普通直连路径被无关字段干扰。
        read_from = [str(item).strip() for item in step["read_from"] if str(item).strip()]

    all_outputs = state.get("step_outputs") or {}
    if not isinstance(all_outputs, dict):
        all_outputs = {}
    upstream_outputs: dict[str, StepOutput] = {}
    for dependency_id in read_from:
        # 只注入显式声明的上游步骤输出，实现“最小上下文传递”。
        payload = all_outputs.get(dependency_id)
        if isinstance(payload, dict):
            upstream_outputs[dependency_id] = payload

    user_input = str(state.get("user_input") or "").strip() if include_user_input else ""
    history_messages = (
        list(state.get("history_messages") or []) if include_chat_history else []
    )

    return {
        "step": step,
        "step_id": step_id,
        "task_description": task_description,
        "coordinator_mode": coordinator_mode,
        "read_from": read_from,
        "upstream_outputs": upstream_outputs,
        "include_user_input": include_user_input,
        "include_chat_history": include_chat_history,
        "user_input": user_input,
        "history_messages": history_messages,
        "final_output": bool(step.get("final_output")),
    }


def build_instruction_text(runtime: dict[str, Any]) -> str:
    """
    将任务描述、可读上下文与上游产出合并为 instruction 文本。
    """
    # 统一生成 instruction 文本，降低各节点拼 prompt 的重复代码。
    sections = [f"任务描述：{runtime.get('task_description') or ''}"]

    user_input = runtime.get("user_input")
    if isinstance(user_input, str) and user_input:
        sections.append(f"用户输入：{user_input}")

    history_messages = runtime.get("history_messages")
    if isinstance(history_messages, list) and history_messages:
        sections.append(
            "历史对话：\n" + json.dumps(history_messages, ensure_ascii=False)
        )

    upstream_outputs = runtime.get("upstream_outputs")
    if isinstance(upstream_outputs, dict) and upstream_outputs:
        sections.append(
            "上游步骤输出：\n" + json.dumps(upstream_outputs, ensure_ascii=False)
        )

    return "\n\n".join(sections)


def build_step_output_update(
        runtime: dict[str, Any],
        *,
        node_name: str,
        status: str,
        text: str = "",
        output: dict[str, Any] | None = None,
        error: str | None = None,
) -> dict[str, Any]:
    """
    基于当前 step_id 构建 step_outputs 增量更新。
    """
    step_id = str(runtime.get("step_id") or "").strip()
    if not step_id:
        # 没有 step_id（例如 gateway 直连模式）就不写 step_outputs，保持兼容。
        return {}

    # 统一输出结构，供 planner 下一轮判断 completed/failed/skipped。
    payload: StepOutput = {
        "step_id": step_id,
        "node_name": node_name,
        "status": status,  # type: ignore[typeddict-item]
        "text": text,
        "output": output or {},
    }
    if error:
        payload["error"] = error

    return {"step_outputs": {step_id: payload}}
