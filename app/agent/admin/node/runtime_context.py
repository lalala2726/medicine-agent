from __future__ import annotations

import json
from typing import Any

from app.agent.admin.agent_state import AgentState, PlanStep, StepFailurePolicy, StepOutput
from app.agent.admin.history_utils import history_to_role_dicts

_DEFAULT_FAILURE_POLICY: StepFailurePolicy = {
    "mode": "hybrid",
    "error_marker_prefix": "__ERROR__:",
    "tool_error_counting": "consecutive",
    "max_tool_errors": 2,
    "strict_data_quality": True,
}

_STRICT_DATA_QUALITY_INSTRUCTION = (
    "\n\n失败策略要求："
    "\n- 当工具调用连续失败或关键数据不可信/不完整时，必须以 '__ERROR__:' 前缀输出错误原因。"
    "\n- 若能确认数据可靠，再输出正常结果。"
)


def _is_coordinator_mode(state: AgentState | dict[str, Any]) -> bool:
    """
    判断当前运行状态是否处于 coordinator 编排模式。

    Args:
        state: 节点执行时的状态对象，需包含 `routing.route_target` 字段。

    Returns:
        当 `routing.route_target == "coordinator_agent"` 时返回 True，否则返回 False。
    """
    # 只有 coordinator 编排路径，才启用“按步骤配置注入上下文”的严格模式。
    routing = state.get("routing") or {}
    return routing.get("route_target") == "coordinator_agent"


def get_current_step(
        state: AgentState | dict[str, Any],
        node_name: str,
) -> dict | dict[Any, Any]:
    """
    获取当前节点在本轮调度中的步骤定义。

    Args:
        state: 节点执行时的状态对象，读取 `routing.current_step_map`。
        node_name: 当前节点名（如 `order_agent`）。

    Returns:
        当前节点对应的步骤配置字典；若不存在或结构非法则返回空字典。
    """
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
    构建当前节点执行所需的标准运行时上下文。

    Args:
        state: 节点执行时的状态对象。
        node_name: 当前节点名（如 `order_agent`）。
        default_task_description: 当步骤未提供 `task_description` 时使用的默认任务描述。

    Returns:
        包含步骤定义、上下文注入开关、上游输出、用户输入、历史消息、失败策略等字段的运行时字典。
    """
    step = get_current_step(state, node_name)
    coordinator_mode = _is_coordinator_mode(state)
    raw_history_messages = list(state.get("history_messages") or [])

    raw_step_id = step.get("step_id")
    step_id = str(raw_step_id).strip() if isinstance(raw_step_id, str) else ""

    task_description = str(step.get("task_description") or "").strip()
    if not task_description:
        task_description = default_task_description

    # 设计约束：
    # - coordinator 模式下默认不读 user/history，必须通过步骤开关显式开启；
    # - 非 coordinator 模式默认可读 history，且有 history 时不再重复注入 user_input。
    if coordinator_mode:
        include_user_input = bool(step.get("include_user_input"))
        include_chat_history = bool(step.get("include_chat_history"))
    else:
        include_chat_history = True
        include_user_input = not raw_history_messages

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
    history_messages = list(raw_history_messages) if include_chat_history else []
    history_messages_serialized = (
        history_to_role_dicts(history_messages) if include_chat_history else []
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
        "history_messages_serialized": history_messages_serialized,
        "final_output": bool(step.get("final_output")),
        "failure_policy": resolve_failure_policy(step),
    }


def resolve_failure_policy(step: PlanStep | dict[str, Any] | None) -> StepFailurePolicy:
    """
    解析步骤级失败策略并补齐默认值。

    Args:
        step: 步骤定义对象，可能包含 `failure_policy` 字段；为 None 时使用默认策略。

    Returns:
        归一化后的步骤失败策略对象。
    """
    raw = (step or {}).get("failure_policy")
    if not isinstance(raw, dict):
        return dict(_DEFAULT_FAILURE_POLICY)

    mode = str(raw.get("mode", _DEFAULT_FAILURE_POLICY["mode"])).strip().lower()
    if mode not in {"hybrid", "marker_only", "tool_only"}:
        mode = str(_DEFAULT_FAILURE_POLICY["mode"])

    error_marker_prefix = str(
        raw.get("error_marker_prefix", _DEFAULT_FAILURE_POLICY["error_marker_prefix"])
    ).strip()
    if not error_marker_prefix:
        error_marker_prefix = str(_DEFAULT_FAILURE_POLICY["error_marker_prefix"])

    tool_error_counting = str(
        raw.get(
            "tool_error_counting",
            _DEFAULT_FAILURE_POLICY["tool_error_counting"],
        )
    ).strip().lower()
    if tool_error_counting not in {"consecutive", "total"}:
        tool_error_counting = str(_DEFAULT_FAILURE_POLICY["tool_error_counting"])

    try:
        max_tool_errors = int(raw.get("max_tool_errors", _DEFAULT_FAILURE_POLICY["max_tool_errors"]))
    except (TypeError, ValueError):
        max_tool_errors = int(_DEFAULT_FAILURE_POLICY["max_tool_errors"])
    if max_tool_errors < 1 or max_tool_errors > 5:
        max_tool_errors = int(_DEFAULT_FAILURE_POLICY["max_tool_errors"])

    raw_strict_data_quality = raw.get(
        "strict_data_quality", _DEFAULT_FAILURE_POLICY["strict_data_quality"]
    )
    if isinstance(raw_strict_data_quality, str):
        strict_data_quality = raw_strict_data_quality.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    else:
        strict_data_quality = bool(raw_strict_data_quality)

    return {
        "mode": mode,  # type: ignore[typeddict-item]
        "error_marker_prefix": error_marker_prefix,
        "tool_error_counting": tool_error_counting,  # type: ignore[typeddict-item]
        "max_tool_errors": max_tool_errors,
        "strict_data_quality": strict_data_quality,
    }


def build_instruction_text(runtime: dict[str, Any]) -> str:
    """
    将任务描述、可读上下文与上游产出合并为 instruction 文本。

    Args:
        runtime: `build_step_runtime` 生成的运行时上下文字典。

    Returns:
        提供给模型的 instruction 文本（包含任务描述、用户输入、历史对话、上游步骤输出等）。
    """
    # 统一生成 instruction 文本，降低各节点拼 prompt 的重复代码。
    sections = [f"任务描述：{runtime.get('task_description') or ''}"]

    user_input = runtime.get("user_input")
    if isinstance(user_input, str) and user_input:
        sections.append(f"用户输入：{user_input}")

    history_messages = runtime.get("history_messages_serialized")
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


def build_instruction_with_failure_policy(runtime: dict[str, Any]) -> str:
    """
    构建包含失败策略提示的 instruction 文本。

    规则：
    - 基础内容沿用 `build_instruction_text`（任务描述、上下文、上游输出）
    - 当 strict_data_quality=true 时，追加统一失败策略提示

    Args:
        runtime: `build_step_runtime` 生成的运行时上下文字典。

    Returns:
        拼接失败策略提示后的 instruction 文本。
    """
    instruction = build_instruction_text(runtime)
    failure_policy = runtime.get("failure_policy") or {}
    if bool(failure_policy.get("strict_data_quality", True)):
        instruction += _STRICT_DATA_QUALITY_INSTRUCTION
    return instruction
