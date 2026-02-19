from __future__ import annotations

from typing import Any, Literal

from app.agent.admin.agent_state import (
    AgentState,
    FallbackContext,
    FallbackFailedStep,
    FallbackPartialResult,
    PlanStep,
    StepFailurePolicy,
)

# coordinator 计划生成时按难度选择模型。
_COORDINATOR_MODEL_BY_DIFFICULTY = {
    "simple": "qwen-flash",
    "medium": "qwen-plus",
    "complex": "qwen-max",
}

# plan 审核时允许的业务节点。
_PLAN_ALLOWED_NODES = {
    "order_agent",
    "excel_agent",
    "chart_agent",
    "summary_agent",
    "product_agent",
}
_MAX_PLAN_STEPS_BY_DIFFICULTY = {
    "simple": 1,
    "medium": 3,
    "complex": 6,
}

FailureMode = Literal["hybrid", "marker_only", "tool_only"]
ToolErrorCounting = Literal["consecutive", "total"]

_DEFAULT_FAILURE_MODE: FailureMode = "hybrid"
_DEFAULT_ERROR_MARKER_PREFIX = "__ERROR__:"
_DEFAULT_TOOL_ERROR_COUNTING: ToolErrorCounting = "consecutive"
_DEFAULT_MAX_TOOL_ERRORS = 2
_DEFAULT_STRICT_DATA_QUALITY = True
_STEP_FAILURE_POLICY_DEFAULT: StepFailurePolicy = {
    "mode": _DEFAULT_FAILURE_MODE,
    "error_marker_prefix": _DEFAULT_ERROR_MARKER_PREFIX,
    "tool_error_counting": _DEFAULT_TOOL_ERROR_COUNTING,
    "max_tool_errors": _DEFAULT_MAX_TOOL_ERRORS,
    "strict_data_quality": _DEFAULT_STRICT_DATA_QUALITY,
}

# planner 调度时允许下发执行的节点。
EXECUTION_NODES = (
    "order_agent",
    "excel_agent",
    "chart_agent",
    "summary_agent",
    "product_agent",
)


def _empty_failure_policy() -> StepFailurePolicy:
    """
    返回空失败策略对象（用于校验失败分支）。

    Args:
        无。

    Returns:
        StepFailurePolicy: 空字典形式的失败策略。
    """
    return {}


def _default_failure_policy() -> StepFailurePolicy:
    """
    返回系统默认失败策略。

    Args:
        无。

    Returns:
        StepFailurePolicy: 包含默认 mode、阈值与哨兵配置的策略对象。
    """
    return {
        "mode": _DEFAULT_FAILURE_MODE,
        "error_marker_prefix": _DEFAULT_ERROR_MARKER_PREFIX,
        "tool_error_counting": _DEFAULT_TOOL_ERROR_COUNTING,
        "max_tool_errors": _DEFAULT_MAX_TOOL_ERRORS,
        "strict_data_quality": _DEFAULT_STRICT_DATA_QUALITY,
    }


def _normalize_failure_policy(raw: Any) -> tuple[StepFailurePolicy, str]:
    """
    归一化并校验步骤级失败策略。

    Args:
        raw: 原始 failure_policy 输入。

    Returns:
        二元组 `(policy, reason)`：
        - policy: 归一化后的策略（失败时为空字典）
        - reason: 校验结果，`ok` 表示通过，否则为错误原因。
    """
    if raw is None:
        return _default_failure_policy(), "ok"
    if not isinstance(raw, dict):
        return _empty_failure_policy(), "failure_policy 必须是对象。"

    raw_mode = str(raw.get("mode", _DEFAULT_FAILURE_MODE)).strip().lower()
    if raw_mode == "hybrid":
        mode: FailureMode = "hybrid"
    elif raw_mode == "marker_only":
        mode = "marker_only"
    elif raw_mode == "tool_only":
        mode = "tool_only"
    else:
        return _empty_failure_policy(), f"failure_policy.mode 非法: {raw_mode}。"

    error_marker_prefix = str(
        raw.get(
            "error_marker_prefix",
            _DEFAULT_ERROR_MARKER_PREFIX,
        )
    ).strip()
    if not error_marker_prefix:
        return _empty_failure_policy(), "failure_policy.error_marker_prefix 不能为空。"

    raw_tool_error_counting = str(
        raw.get(
            "tool_error_counting",
            _DEFAULT_TOOL_ERROR_COUNTING,
        )
    ).strip().lower()
    if raw_tool_error_counting == "consecutive":
        tool_error_counting: ToolErrorCounting = "consecutive"
    elif raw_tool_error_counting == "total":
        tool_error_counting = "total"
    else:
        return _empty_failure_policy(), (
            f"failure_policy.tool_error_counting 非法: {raw_tool_error_counting}。"
        )

    raw_max_tool_errors = raw.get("max_tool_errors", _DEFAULT_MAX_TOOL_ERRORS)
    try:
        max_tool_errors = int(raw_max_tool_errors)
    except (TypeError, ValueError):
        return _empty_failure_policy(), "failure_policy.max_tool_errors 必须是整数。"
    if max_tool_errors < 1 or max_tool_errors > 5:
        return _empty_failure_policy(), "failure_policy.max_tool_errors 必须在 1..5 范围内。"

    strict_data_quality = _normalize_bool(
        raw.get(
            "strict_data_quality",
            _DEFAULT_STRICT_DATA_QUALITY,
        ),
        default=_DEFAULT_STRICT_DATA_QUALITY,
    )

    policy: StepFailurePolicy = {
        "mode": mode,
        "error_marker_prefix": error_marker_prefix,
        "tool_error_counting": tool_error_counting,
        "max_tool_errors": max_tool_errors,
        "strict_data_quality": strict_data_quality,
    }
    return policy, "ok"


def select_model_by_difficulty(difficulty: str) -> str:
    """
    根据任务复杂度选择 coordinator 规划阶段使用的模型名称。

    Args:
        difficulty: 网关或上游传入的复杂度标识，期望值为
            `simple` / `medium` / `complex`（大小写不敏感）。

    Returns:
        与复杂度对应的模型名。若值为空或非法，默认返回 `simple` 对应模型。
    """
    key = str(difficulty or "simple").strip().lower()
    return _COORDINATOR_MODEL_BY_DIFFICULTY.get(
        key, _COORDINATOR_MODEL_BY_DIFFICULTY["simple"]
    )


def _difficulty_step_limit(difficulty: str) -> int:
    """
    获取指定复杂度允许的最大计划步骤数。

    Args:
        difficulty: 复杂度标识，通常来自 routing.difficulty。

    Returns:
        该复杂度下的最大步骤数限制。若复杂度不合法，返回 `medium` 的默认限制。
    """
    key = str(difficulty or "medium").strip().lower()
    return _MAX_PLAN_STEPS_BY_DIFFICULTY.get(
        key, _MAX_PLAN_STEPS_BY_DIFFICULTY["medium"]
    )


def _normalize_bool(value: Any, default: bool = False) -> bool:
    """
    将任意输入归一化为布尔值。

    支持常见字符串布尔表示，如 `"true"`/`"false"`、`"1"`/`"0"`。
    无法识别时回退到 `default`。

    Args:
        value: 任意待转换输入。
        default: 兜底值。

    Returns:
        归一化后的布尔值。
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _normalize_str_list(value: Any) -> list[str]:
    """
    将任意输入归一化为字符串列表。

    仅当输入是 list 时进行处理，元素会被转成字符串并去除首尾空白。
    空字符串元素会被过滤。

    Args:
        value: 待归一化输入。

    Returns:
        清洗后的字符串数组；输入不是 list 时返回空数组。
    """
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def _normalize_plan(plan: Any) -> list[dict[str, Any]]:
    """
    对原始 plan 输入做基础结构清洗。

    Args:
        plan: 可能来自 LLM JSON 的 `plan` 字段。

    Returns:
        仅保留字典元素的步骤数组；如果输入不是 list，返回空数组。
    """
    if not isinstance(plan, list):
        return []
    return [item for item in plan if isinstance(item, dict)]


def _has_cycle(graph: dict[str, list[str]]) -> bool:
    """
    检测依赖图是否存在有向环。

    Args:
        graph: 邻接表表示的依赖图，键为步骤 ID，值为其依赖的父步骤 ID 列表。

    Returns:
        若存在循环依赖返回 True，否则返回 False。
    """
    visited: set[str] = set()
    visiting: set[str] = set()

    def dfs(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False

        visiting.add(node)
        for parent in graph.get(node, []):
            if dfs(parent):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    for node_id in graph:
        if dfs(node_id):
            return True
    return False


def _collect_ancestors(
        node_id: str,
        graph: dict[str, list[str]],
        memo: dict[str, set[str]],
) -> set[str]:
    """
    收集某个步骤的全部可达上游祖先步骤。

    使用 DFS + 记忆化缓存，避免重复计算，提高多步骤校验性能。

    Args:
        node_id: 目标步骤 ID。
        graph: 依赖图（step_id -> 依赖列表）。
        memo: 记忆化缓存，key 为 step_id，value 为其祖先集合。

    Returns:
        `node_id` 的所有上游祖先 step_id 集合。
    """
    if node_id in memo:
        return memo[node_id]

    ancestors: set[str] = set()
    for parent in graph.get(node_id, []):
        ancestors.add(parent)
        ancestors.update(_collect_ancestors(parent, graph, memo))
    memo[node_id] = ancestors
    return ancestors


def review_plan(plan: Any, difficulty: str) -> tuple[bool, list[PlanStep], str]:
    """
    审核并规范化 coordinator 生成的 DAG 计划。

    主要校验：
    - 步骤结构与必填字段完整性
    - step_id 唯一性
    - 节点白名单
    - required/optional 依赖引用合法性
    - DAG 无环（required ∪ optional）
    - read_from 仅可读取可达上游
    - failure_policy 字段合法性与默认补全
    - final_output 唯一且为终点
    - 步骤数不超过复杂度上限

    Args:
        plan: LLM 返回的原始计划结构，期望为步骤对象数组。
        difficulty: 当前请求复杂度，用于限制步骤总数。

    Returns:
        三元组 `(is_valid, normalized_plan, reason)`：
        - is_valid: 是否通过校验。
        - normalized_plan: 校验通过后的规范化步骤数组；失败时为空列表。
        - reason: 校验结果说明；通过时为 `"ok"`，失败时为可回显给模型的原因。
    """
    normalized_items = _normalize_plan(plan)
    if not normalized_items:
        return False, [], "plan 为空或结构非法，至少需要一个可执行步骤。"

    max_steps = _difficulty_step_limit(difficulty)
    if len(normalized_items) > max_steps:
        return (
            False,
            [],
            f"plan 复杂度过高（步骤数 {len(normalized_items)} 超过 {difficulty} 允许上限 {max_steps}）。",
        )

    seen_step_ids: set[str] = set()
    normalized_plan: list[PlanStep] = []
    final_output_step_ids: list[str] = []

    for index, raw_step in enumerate(normalized_items, start=1):
        required_keys = {
            "step_id",
            "node_name",
            "task_description",
            "required_depends_on",
            "optional_depends_on",
            "read_from",
            "include_user_input",
            "include_chat_history",
            "final_output",
        }
        missing_keys = [key for key in required_keys if key not in raw_step]
        if missing_keys:
            return (
                False,
                [],
                f"第{index}个步骤缺少必要字段: {missing_keys}。",
            )

        step_id = str(raw_step.get("step_id") or "").strip()
        if not step_id:
            return False, [], f"第{index}个步骤缺少有效 step_id。"
        if step_id in seen_step_ids:
            return False, [], f"step_id 重复: {step_id}。"
        seen_step_ids.add(step_id)

        node_name = str(raw_step.get("node_name") or "").strip()
        if not node_name:
            return False, [], f"步骤 {step_id} 缺少 node_name。"
        if node_name == "coordinator_agent":
            return False, [], "plan 中出现 coordinator_agent，会造成循环依赖。"
        if node_name not in _PLAN_ALLOWED_NODES:
            return False, [], f"节点 {node_name} 未实现，不可出现在 plan 中。"

        task_description = str(raw_step.get("task_description") or "").strip()
        if not task_description:
            return False, [], f"步骤 {step_id} 缺少有效 task_description。"

        required_depends_on = _normalize_str_list(raw_step.get("required_depends_on"))
        optional_depends_on = _normalize_str_list(raw_step.get("optional_depends_on"))
        read_from = _normalize_str_list(raw_step.get("read_from"))
        if step_id in required_depends_on:
            return False, [], f"步骤 {step_id} 不能在 required_depends_on 中依赖自身。"
        if step_id in optional_depends_on:
            return False, [], f"步骤 {step_id} 不能在 optional_depends_on 中依赖自身。"
        if step_id in read_from:
            return False, [], f"步骤 {step_id} 不能读取自身。"
        overlap = set(required_depends_on) & set(optional_depends_on)
        if overlap:
            return (
                False,
                [],
                f"步骤 {step_id} 的 required_depends_on 与 optional_depends_on 存在重复依赖: {sorted(overlap)}。",
            )

        include_user_input = _normalize_bool(
            raw_step.get("include_user_input"), default=False
        )
        include_chat_history = _normalize_bool(
            raw_step.get("include_chat_history"), default=False
        )
        final_output = _normalize_bool(raw_step.get("final_output"), default=False)
        if final_output:
            final_output_step_ids.append(step_id)
        failure_policy, policy_reason = _normalize_failure_policy(
            raw_step.get("failure_policy")
        )
        if policy_reason != "ok":
            return False, [], f"步骤 {step_id} 的 {policy_reason}"

        normalized_plan.append(
            {
                "step_id": step_id,
                "node_name": node_name,
                "task_description": task_description,
                "required_depends_on": required_depends_on,
                "optional_depends_on": optional_depends_on,
                "read_from": read_from,
                "include_user_input": include_user_input,
                "include_chat_history": include_chat_history,
                "final_output": final_output,
                "failure_policy": failure_policy,
            }
        )

    if len(final_output_step_ids) != 1:
        return False, [], "final_output=true 必须且仅能出现一次。"

    step_id_set = {step["step_id"] for step in normalized_plan if step.get("step_id")}
    dependency_graph: dict[str, list[str]] = {}
    for step in normalized_plan:
        step_id = str(step["step_id"])
        required_depends_on = list(step.get("required_depends_on") or [])
        optional_depends_on = list(step.get("optional_depends_on") or [])
        read_from = list(step.get("read_from") or [])

        for dependency_id in required_depends_on:
            if dependency_id not in step_id_set:
                return (
                    False,
                    [],
                    f"步骤 {step_id} 的 required_depends_on 引用了不存在的步骤 {dependency_id}。",
                )
        for dependency_id in optional_depends_on:
            if dependency_id not in step_id_set:
                return (
                    False,
                    [],
                    f"步骤 {step_id} 的 optional_depends_on 引用了不存在的步骤 {dependency_id}。",
                )
        for read_id in read_from:
            if read_id not in step_id_set:
                return False, [], f"步骤 {step_id} 的 read_from 引用了不存在的步骤 {read_id}。"

        dependency_graph[step_id] = required_depends_on + optional_depends_on

    if _has_cycle(dependency_graph):
        return False, [], "plan 存在循环依赖。"

    ancestors_memo: dict[str, set[str]] = {}
    for step in normalized_plan:
        step_id = str(step["step_id"])
        allowed_read_sources = _collect_ancestors(step_id, dependency_graph, ancestors_memo)
        for read_id in step.get("read_from") or []:
            if read_id not in allowed_read_sources:
                return (
                    False,
                    [],
                    f"步骤 {step_id} 的 read_from 包含不可达上游步骤 {read_id}。",
                )

    final_step_id = final_output_step_ids[0]
    for step in normalized_plan:
        if final_step_id in (step.get("required_depends_on") or []) or final_step_id in (
                step.get("optional_depends_on") or []
        ):
            return (
                False,
                [],
                f"final_output 步骤 {final_step_id} 不能被其他步骤依赖。",
            )

    return True, normalized_plan, "ok"


def build_retry_feedback(reason: str) -> str:
    """
    生成 coordinator 重试提示文本。

    用于在 plan 审核失败后把失败原因反馈给模型，指导其按新 schema 重新生成。

    Args:
        reason: 本轮审核失败原因。

    Returns:
        可直接追加到重试消息中的提示文本。
    """
    return (
        "上一次生成的 plan 未通过系统校验，原因："
        f"{reason}。\n"
        "请重新生成完整 JSON（仅包含 plan），并严格遵守新 DAG 字段："
        "step_id/required_depends_on/optional_depends_on/read_from/"
        "include_user_input/include_chat_history/final_output/"
        "failure_policy。"
    )


def _normalize_plan_steps(plan: Any) -> list[PlanStep]:
    """
    对运行态计划做轻量清洗，过滤无效步骤。

    Args:
        plan: 当前状态中的 plan。

    Returns:
        仅保留字典类型且包含非空 step_id 的步骤列表。
    """
    if not isinstance(plan, list):
        return []
    normalized: list[PlanStep] = []
    for item in plan:
        if not isinstance(item, dict):
            continue
        step_id = str(item.get("step_id") or "").strip()
        if not step_id:
            continue
        step: PlanStep = {"step_id": step_id}
        if "node_name" in item:
            step["node_name"] = item.get("node_name")
        if "required_depends_on" in item:
            step["required_depends_on"] = item.get("required_depends_on")
        if "optional_depends_on" in item:
            step["optional_depends_on"] = item.get("optional_depends_on")
        if "read_from" in item:
            step["read_from"] = item.get("read_from")
        if "task_description" in item:
            step["task_description"] = item.get("task_description")
        if "include_user_input" in item:
            step["include_user_input"] = item.get("include_user_input")
        if "include_chat_history" in item:
            step["include_chat_history"] = item.get("include_chat_history")
        if "final_output" in item:
            step["final_output"] = item.get("final_output")
        if "failure_policy" in item:
            step["failure_policy"] = item.get("failure_policy")
        normalized.append(step)
    return normalized


def _extract_status(step_outputs: dict[str, Any]) -> dict[str, str]:
    """
    从 step_outputs 中提取调度所需的终态状态。

    Args:
        step_outputs: 节点执行后回写的步骤输出字典。

    Returns:
        `{step_id: status}` 映射，仅保留 `completed/failed/skipped` 三种状态。
    """
    statuses: dict[str, str] = {}
    for step_id, payload in step_outputs.items():
        if not isinstance(payload, dict):
            continue
        status = str(payload.get("status") or "").strip().lower()
        if status in {"completed", "failed", "skipped"}:
            statuses[step_id] = status
    return statuses


def _build_skipped_step_output(
        step: PlanStep,
        *,
        failed_dependencies: list[str],
) -> dict[str, Any]:
    """
    构造被阻断步骤的标准 `skipped` 输出。

    Args:
        step: 被阻断步骤的计划定义。
        failed_dependencies: 导致阻断的依赖步骤 ID 列表。

    Returns:
        符合统一输出结构的步骤结果对象，状态固定为 `skipped`。
    """
    step_id = str(step.get("step_id") or "")
    node_name = str(step.get("node_name") or "")
    message = (
        f"步骤 {step_id} 被阻断：依赖步骤 {failed_dependencies} 执行失败或已跳过。"
    )
    return {
        "step_id": step_id,
        "node_name": node_name,
        "status": "skipped",
        "text": message,
        "output": {},
        "error": message,
    }


def _build_terminated_step_output(step: PlanStep, *, reason: str) -> dict[str, Any]:
    """
    构造 fallback 终止时的 skipped 输出。

    Args:
        step: 步骤定义。
        reason: 终止原因说明。

    Returns:
        dict[str, Any]: 标准化 skipped 步骤输出对象。
    """
    step_id = str(step.get("step_id") or "")
    node_name = str(step.get("node_name") or "")
    message = f"步骤 {step_id} 未执行：{reason}"
    return {
        "step_id": step_id,
        "node_name": node_name,
        "status": "skipped",
        "text": message,
        "output": {},
        "error": message,
    }


def _build_fallback_context(
        *,
        ordered_step_ids: list[str],
        step_by_id: dict[str, PlanStep],
        status_by_id: dict[str, str],
        step_outputs: dict[str, Any],
        final_step_id: str,
) -> FallbackContext:
    """
    构造 planner -> chat 兜底输出所需上下文。

    Args:
        ordered_step_ids: 按执行拓扑排序的步骤 ID 列表。
        step_by_id: step_id 到步骤定义的映射。
        status_by_id: step_id 到执行状态的映射。
        step_outputs: 各步骤输出载荷。
        final_step_id: 标记为 final_output 的步骤 ID。

    Returns:
        FallbackContext: fallback 聊天节点可消费的上下文对象。
    """
    failed_steps: list[FallbackFailedStep] = []
    partial_results: list[FallbackPartialResult] = []
    for step_id in ordered_step_ids:
        status = status_by_id.get(step_id)
        if not status:
            continue
        payload = step_outputs.get(step_id) or {}
        step = step_by_id.get(step_id) or {}
        node_name = str(payload.get("node_name") or step.get("node_name") or "")
        if status == "failed":
            error = str(payload.get("error") or payload.get("text") or "").strip()
            failed_step: FallbackFailedStep = {
                "step_id": step_id,
                "node_name": node_name,
                "status": "failed",
                "error": error,
            }
            failed_steps.append(failed_step)
        elif status == "skipped":
            error = str(payload.get("error") or payload.get("text") or "").strip()
            failed_step: FallbackFailedStep = {
                "step_id": step_id,
                "node_name": node_name,
                "status": "skipped",
                "error": error,
            }
            failed_steps.append(failed_step)
        elif status == "completed":
            text = str(payload.get("text") or "").strip()
            if text:
                partial_result: FallbackPartialResult = {
                    "step_id": step_id,
                    "node_name": node_name,
                    "text": text,
                }
                partial_results.append(partial_result)

    final_status = status_by_id.get(final_step_id)
    if final_status == "failed":
        reason_text = f"最终步骤 {final_step_id} 执行失败。"
    elif final_status == "skipped":
        reason_text = f"最终步骤 {final_step_id} 因必选依赖失败不可达。"
    else:
        reason_text = f"最终步骤 {final_step_id} 不可达。"

    context: FallbackContext = {
        "trigger": "final_output_unreachable",
        "final_step_id": final_step_id,
        "failed_steps": failed_steps,
        "partial_results": partial_results,
        "reason_text": reason_text,
    }
    return context


def compute_planner_update(state: AgentState) -> dict[str, Any]:
    """
    计算 planner 每一轮的路由更新结果（纯规则函数，无图执行副作用）。

    规则要点：
    1. 只根据 state.plan + state.step_outputs 推导下一轮可执行步骤。
    2. 依赖失败会向下游传播阻断，并为阻断步骤生成 skipped 输出。
    3. 同一轮如果出现相同 node_name 的多个 ready 步骤，只放行一个，避免同节点并发。
    4. 通过 routing.current_step_map 把“当前轮步骤配置”下发给具体节点读取。

    Args:
        state: 当前 Agent 运行状态，至少包含 `plan`、`routing`、`step_outputs`。

    Returns:
        更新字典，最少包含 `routing`；当有新阻断步骤时会附带 `step_outputs` 增量：
        - `routing.next_nodes`: 下一轮应执行的节点名列表
        - `routing.next_step_ids`: 与 `next_nodes` 对应的 step_id 列表
        - `routing.current_step_map`: 当前轮 node_name -> step 配置
        - `routing.completed_step_ids`: 已完成步骤集合（按计划顺序）
        - `routing.blocked_step_ids`: 被阻断/已跳过步骤集合（按计划顺序）
        - `routing.is_final_stage`: 当前轮是否包含 final_output 步骤
    """
    plan_steps = _normalize_plan_steps(state.get("plan"))
    routing = dict(state.get("routing") or {})

    # 无可执行计划时，输出空调度状态并尽快结束本轮。
    if not plan_steps:
        routing["next_nodes"] = []
        routing["next_step_ids"] = []
        routing["current_step_ids"] = []
        routing["current_step_map"] = {}
        routing["is_final_stage"] = False
        routing.pop("fallback_context", None)
        return {"routing": routing}

    step_by_id: dict[str, PlanStep] = {}
    ordered_step_ids: list[str] = []
    final_step_id = ""
    for step in plan_steps:
        step_id = str(step.get("step_id") or "").strip()
        if step_id and step_id not in step_by_id:
            step_by_id[step_id] = step
            ordered_step_ids.append(step_id)
            if not final_step_id and bool(step.get("final_output")):
                final_step_id = step_id

    raw_step_outputs = state.get("step_outputs") or {}
    step_outputs = raw_step_outputs if isinstance(raw_step_outputs, dict) else {}
    status_by_id = _extract_status(step_outputs)

    failed_or_skipped_ids = {
        sid for sid, status in status_by_id.items() if status in {"failed", "skipped"}
    }
    terminal_ids = set(status_by_id.keys())

    blocked_ids: set[str] = set()
    blocked_reasons: dict[str, list[str]] = {}
    changed = True
    # 传播阻断仅沿 required_depends_on：必选依赖失败才会阻断下游。
    while changed:
        changed = False
        failed_pool = failed_or_skipped_ids | blocked_ids
        for step_id in ordered_step_ids:
            if step_id in terminal_ids or step_id in blocked_ids:
                continue
            step = step_by_id.get(step_id) or {}
            required_deps = [
                dep for dep in (step.get("required_depends_on") or []) if isinstance(dep, str)
            ]
            failed_dependencies = [dep for dep in required_deps if dep in failed_pool]
            if failed_dependencies:
                blocked_ids.add(step_id)
                blocked_reasons[step_id] = failed_dependencies
                changed = True

    blocked_updates: dict[str, Any] = {}
    for blocked_step_id in blocked_ids:
        if blocked_step_id in status_by_id:
            continue
        step = step_by_id.get(blocked_step_id) or {}
        blocked_updates[blocked_step_id] = _build_skipped_step_output(
            step,
            failed_dependencies=blocked_reasons.get(blocked_step_id, []),
        )

    # 合并本轮新阻断后的状态，供 optional 终态等待和 fallback 判定。
    status_after_block = dict(status_by_id)
    for blocked_step_id in blocked_updates:
        status_after_block[blocked_step_id] = "skipped"
    completed_after_block = {
        sid for sid, status in status_after_block.items() if status == "completed"
    }
    terminal_after_block = set(status_after_block.keys())

    # final 不可达/失败时，立即切换到 chat fallback。
    final_status = status_after_block.get(final_step_id) if final_step_id else None
    should_fallback = final_status in {"failed", "skipped"}
    fallback_reason = "已进入兜底回答流程，当前步骤不再执行。"

    fallback_updates: dict[str, Any] = {}
    if should_fallback:
        for step_id in ordered_step_ids:
            if step_id in terminal_after_block:
                continue
            step = step_by_id.get(step_id) or {}
            fallback_updates[step_id] = _build_terminated_step_output(
                step,
                reason=fallback_reason,
            )
        step_outputs_for_fallback = dict(step_outputs)
        step_outputs_for_fallback.update(blocked_updates)
        step_outputs_for_fallback.update(fallback_updates)
        status_for_fallback = _extract_status(step_outputs_for_fallback)
        routing["fallback_context"] = _build_fallback_context(
            ordered_step_ids=ordered_step_ids,
            step_by_id=step_by_id,
            status_by_id=status_for_fallback,
            step_outputs=step_outputs_for_fallback,
            final_step_id=final_step_id,
        )
        routing["next_nodes"] = ["chat_agent"]
        routing["next_step_ids"] = []
        routing["current_step_ids"] = []
        routing["current_step_map"] = {}
        routing["completed_step_ids"] = [
            sid for sid in ordered_step_ids if status_for_fallback.get(sid) == "completed"
        ]
        routing["blocked_step_ids"] = [
            sid for sid in ordered_step_ids if status_for_fallback.get(sid) == "skipped"
        ]
        routing["is_final_stage"] = False
        try:
            stage_index = int(routing.get("stage_index", 0))
        except (TypeError, ValueError):
            stage_index = 0
        routing["stage_index"] = stage_index + 1

        merged_updates = dict(blocked_updates)
        merged_updates.update(fallback_updates)
        result: dict[str, Any] = {"routing": routing}
        if merged_updates:
            result["step_outputs"] = merged_updates
        return result

    ready_steps: list[PlanStep] = []
    used_nodes: set[str] = set()
    # 按计划顺序挑选 ready 步骤，保证执行顺序稳定可预测。
    for step_id in ordered_step_ids:
        if step_id in terminal_after_block:
            continue
        step = step_by_id.get(step_id) or {}
        node_name = str(step.get("node_name") or "").strip()
        if node_name not in EXECUTION_NODES:
            continue

        required_deps = [
            dep for dep in (step.get("required_depends_on") or []) if isinstance(dep, str)
        ]
        optional_deps = [
            dep for dep in (step.get("optional_depends_on") or []) if isinstance(dep, str)
        ]
        if not all(dep in completed_after_block for dep in required_deps):
            continue
        # optional 依赖失败可继续，但必须进入终态后才能执行当前步骤。
        if not all(dep in terminal_after_block for dep in optional_deps):
            continue
        if node_name in used_nodes:
            continue

        used_nodes.add(node_name)
        ready_steps.append(step)

    next_nodes = [str(step.get("node_name")) for step in ready_steps]
    next_step_ids = [str(step.get("step_id")) for step in ready_steps]
    # node_name -> step 配置映射，供节点运行时获取 read_from 等参数。
    current_step_map = {
        str(step.get("node_name")): step for step in ready_steps if step.get("node_name")
    }

    blocked_step_ids = [
        step_id
        for step_id in ordered_step_ids
        if status_after_block.get(step_id) == "skipped"
    ]
    completed_step_ids = [
        step_id for step_id in ordered_step_ids if status_after_block.get(step_id) == "completed"
    ]

    routing.pop("fallback_context", None)
    routing["next_nodes"] = next_nodes
    routing["next_step_ids"] = next_step_ids
    routing["current_step_ids"] = next_step_ids
    routing["current_step_map"] = current_step_map
    routing["completed_step_ids"] = completed_step_ids
    routing["blocked_step_ids"] = blocked_step_ids
    # 当前轮包含 final_output=true 的步骤时，SSE/输出层可直接按最终节点处理。
    routing["is_final_stage"] = any(bool(step.get("final_output")) for step in ready_steps)
    try:
        stage_index = int(routing.get("stage_index", 0))
    except (TypeError, ValueError):
        stage_index = 0
    routing["stage_index"] = stage_index + 1

    result = {"routing": routing}
    if blocked_updates:
        result["step_outputs"] = blocked_updates
    return result
