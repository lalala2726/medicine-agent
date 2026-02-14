from __future__ import annotations

from typing import Any

from app.agent.admin.agent_state import AgentState, PlanStep

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

# planner 调度时允许下发执行的节点。
EXECUTION_NODES = (
    "order_agent",
    "excel_agent",
    "chart_agent",
    "summary_agent",
    "product_agent",
)


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
        graph: 依赖图（step_id -> depends_on 列表）。
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
    - depends_on/read_from 引用合法性
    - DAG 无环
    - read_from 仅可读取可达上游
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

        depends_on = _normalize_str_list(raw_step.get("depends_on"))
        read_from = _normalize_str_list(raw_step.get("read_from"))
        if step_id in depends_on:
            return False, [], f"步骤 {step_id} 不能依赖自身。"
        if step_id in read_from:
            return False, [], f"步骤 {step_id} 不能读取自身。"

        include_user_input = _normalize_bool(
            raw_step.get("include_user_input"), default=False
        )
        include_chat_history = _normalize_bool(
            raw_step.get("include_chat_history"), default=False
        )
        final_output = _normalize_bool(raw_step.get("final_output"), default=False)
        if final_output:
            final_output_step_ids.append(step_id)

        normalized_plan.append(
            {
                "step_id": step_id,
                "node_name": node_name,
                "task_description": task_description,
                "depends_on": depends_on,
                "read_from": read_from,
                "include_user_input": include_user_input,
                "include_chat_history": include_chat_history,
                "final_output": final_output,
            }
        )

    if len(final_output_step_ids) != 1:
        return False, [], "final_output=true 必须且仅能出现一次。"

    step_id_set = {step["step_id"] for step in normalized_plan if step.get("step_id")}
    dependency_graph: dict[str, list[str]] = {}
    for step in normalized_plan:
        step_id = str(step["step_id"])
        depends_on = list(step.get("depends_on") or [])
        read_from = list(step.get("read_from") or [])

        for dependency_id in depends_on:
            if dependency_id not in step_id_set:
                return False, [], f"步骤 {step_id} 的 depends_on 引用了不存在的步骤 {dependency_id}。"
        for read_id in read_from:
            if read_id not in step_id_set:
                return False, [], f"步骤 {step_id} 的 read_from 引用了不存在的步骤 {read_id}。"

        dependency_graph[step_id] = depends_on

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
        if final_step_id in (step.get("depends_on") or []):
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
        "step_id/depends_on/read_from/include_user_input/include_chat_history/final_output。"
    )


def _normalize_plan_steps(plan: list[PlanStep] | None) -> list[PlanStep]:
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
        normalized.append(item)
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
        return {"routing": routing}

    step_by_id: dict[str, PlanStep] = {}
    ordered_step_ids: list[str] = []
    for step in plan_steps:
        step_id = str(step.get("step_id") or "").strip()
        if step_id and step_id not in step_by_id:
            step_by_id[step_id] = step
            ordered_step_ids.append(step_id)

    raw_step_outputs = state.get("step_outputs") or {}
    step_outputs = raw_step_outputs if isinstance(raw_step_outputs, dict) else {}
    status_by_id = _extract_status(step_outputs)

    completed_ids = {sid for sid, status in status_by_id.items() if status == "completed"}
    failed_or_skipped_ids = {
        sid for sid, status in status_by_id.items() if status in {"failed", "skipped"}
    }
    terminal_ids = set(status_by_id.keys())

    blocked_ids: set[str] = set()
    blocked_reasons: dict[str, list[str]] = {}
    changed = True
    # 传播阻断直到收敛：A 失败导致 B 阻断，B 阻断继续导致 C 阻断。
    while changed:
        changed = False
        failed_pool = failed_or_skipped_ids | blocked_ids
        for step_id in ordered_step_ids:
            if step_id in terminal_ids or step_id in blocked_ids:
                continue
            step = step_by_id.get(step_id) or {}
            depends_on = [
                dep for dep in (step.get("depends_on") or []) if isinstance(dep, str)
            ]
            failed_dependencies = [dep for dep in depends_on if dep in failed_pool]
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

    ready_steps: list[PlanStep] = []
    used_nodes: set[str] = set()
    terminal_or_blocked = terminal_ids | blocked_ids
    # 按计划顺序挑选 ready 步骤，保证执行顺序稳定可预测。
    for step_id in ordered_step_ids:
        if step_id in terminal_or_blocked:
            continue
        step = step_by_id.get(step_id) or {}
        node_name = str(step.get("node_name") or "").strip()
        if node_name not in EXECUTION_NODES:
            continue

        depends_on = [dep for dep in (step.get("depends_on") or []) if isinstance(dep, str)]
        if not all(dep in completed_ids for dep in depends_on):
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
        if status_by_id.get(step_id) == "skipped" or step_id in blocked_ids
    ]
    completed_step_ids = [
        step_id for step_id in ordered_step_ids if status_by_id.get(step_id) == "completed"
    ]

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

    result: dict[str, Any] = {"routing": routing}
    if blocked_updates:
        result["step_outputs"] = blocked_updates
    return result
