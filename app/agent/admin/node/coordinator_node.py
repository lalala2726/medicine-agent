import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.admin.agent_state import AgentState, PlanStep
from app.core.assistant_status import status_node
from app.core.langsmith import traceable
from app.core.llm import create_chat_model

_system_prompt = """
你是后台工作流中的 coordinator_agent，负责把用户请求拆解成 DAG 计划（仅规划，不执行）。

可用执行节点（node_name 只能从以下值中选择）：
- order_agent: 订单查询、订单状态与订单信息核验。
- excel_agent: 表格解析、表格整理、Excel 导出。
- chart_agent: 基于已有结构化数据生成图表或统计说明。
- summary_agent: 对多个节点结果做汇总并输出最终结论。
- product_agent: 商品查询、商品详情查询。

注意：
- chat_agent 由 gateway_router 处理，禁止出现在 plan 中。
- 你必须输出新格式，不允许旧的分阶段嵌套数组结构。

输出 JSON 格式（仅输出 JSON，不要额外文本）：
{
  "plan": [
    {
      "step_id": "s1",
      "node_name": "order_agent",
      "task_description": "string",
      "depends_on": [],
      "read_from": [],
      "include_user_input": false,
      "include_chat_history": false,
      "final_output": false
    },
    {
      "step_id": "s2",
      "node_name": "summary_agent",
      "task_description": "string",
      "depends_on": ["s1"],
      "read_from": ["s1"],
      "include_user_input": false,
      "include_chat_history": false,
      "final_output": true
    }
  ]
}

严格规则：
1. 顶层只允许一个键：plan，且 plan 必须是对象数组。
2. 每个步骤必须包含：step_id/node_name/task_description/depends_on/read_from/include_user_input/include_chat_history/final_output。
3. step_id 必须全局唯一。
4. node_name 不得为 coordinator_agent，且必须是可用节点之一。
5. depends_on / read_from 必须是字符串数组，且引用的 step_id 必须存在。
6. read_from 只能读取该步骤“可达上游”步骤，不能读取无依赖关系步骤，也不能读取自身。
7. plan 必须无环。
8. final_output=true 必须且仅能有一个。
9. final_output 对应步骤不能被其他步骤依赖（必须是最终输出端）。
10. 若需求不清晰，输出最小可执行 DAG（至少 1 个步骤，且 final_output 唯一）。
"""

_COORDINATOR_MODEL_BY_DIFFICULTY = {
    "simple": "qwen-flash",
    "medium": "qwen-plus",
    "complex": "qwen-max",
}

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
_PLAN_REVIEW_RETRY_LIMIT = 2


def _select_model_by_difficulty(difficulty: str) -> str:
    # simple/medium/complex -> 不同成本与能力模型，避免简单问题用重模型。
    key = str(difficulty or "simple").strip().lower()
    return _COORDINATOR_MODEL_BY_DIFFICULTY.get(
        key, _COORDINATOR_MODEL_BY_DIFFICULTY["simple"]
    )


def _difficulty_step_limit(difficulty: str) -> int:
    # 难度越高允许的步骤越多，用于防止 plan 无限膨胀。
    key = str(difficulty or "medium").strip().lower()
    return _MAX_PLAN_STEPS_BY_DIFFICULTY.get(
        key, _MAX_PLAN_STEPS_BY_DIFFICULTY["medium"]
    )


def _normalize_bool(value: Any, default: bool = False) -> bool:
    # 容错处理：模型有时会返回 "true"/"false" 字符串，这里统一归一化。
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
    # 容错处理：确保 depends_on/read_from 一定是“去空白后的字符串数组”。
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def _normalize_plan(plan: Any) -> list[dict[str, Any]]:
    # 新版只接受“扁平 steps 数组”；旧的阶段嵌套数组会在这里被直接过滤掉。
    if not isinstance(plan, list):
        return []
    return [item for item in plan if isinstance(item, dict)]


def _has_cycle(graph: dict[str, list[str]]) -> bool:
    # 使用 DFS 检测有向图环路：只要有回边就判定为非法 DAG。
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
    # 计算某节点的全部可达上游，校验 read_from 是否越权读取。
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
    审核协调器生成的 DAG 计划。
    """
    # 第 1 层：结构与规模校验。
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

    # 第 2 层：单步骤字段校验与标准化。
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

    # 第 3 层：图级校验（依赖存在、无环、read_from 上游可达、final_output 终点约束）。
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


def _build_retry_feedback(reason: str) -> str:
    # 把失败原因喂回模型，强制其按新 schema 重新生成。
    return (
        "上一次生成的 plan 未通过系统校验，原因："
        f"{reason}。\n"
        "请重新生成完整 JSON（仅包含 plan），并严格遵守新 DAG 字段："
        "step_id/depends_on/read_from/include_user_input/include_chat_history/final_output。"
    )


@status_node(node="coordinator", start_message="正在规划任务中")
@traceable(name="Coordinator Agent Node", run_type="chain")
def coordinator(state: AgentState) -> dict[str, Any]:
    routing = dict(state.get("routing") or {})
    difficulty = str(routing.get("difficulty") or "medium").strip().lower()
    model_name = _select_model_by_difficulty(difficulty)

    user_input = str(state.get("user_input") or "").strip()
    if not user_input:
        routing["difficulty"] = difficulty
        return {"routing": routing, "plan": []}

    # coordinator 只负责产出可执行计划，不做工具调用。
    llm = create_chat_model(
        model=model_name,
        temperature=0,
        response_format={"type": "json_object"},
    )
    base_messages = [
        SystemMessage(content=_system_prompt),
        HumanMessage(content=f"用户请求：{user_input}"),
    ]

    routing["difficulty"] = difficulty
    reviewed_plan: list[PlanStep] = []
    last_reason = "模型返回内容不是有效 JSON。"

    for attempt in range(_PLAN_REVIEW_RETRY_LIMIT + 1):
        messages = list(base_messages)
        if attempt > 0:
            messages.append(HumanMessage(content=_build_retry_feedback(last_reason)))

        try:
            response = llm.invoke(messages)
            payload = json.loads(str(response.content))
        except Exception:
            last_reason = "模型返回内容不是有效 JSON。"
            continue

        # 审核失败会触发重试；重试上限后返回空 plan，由上层做降级。
        is_valid, candidate_plan, reason = review_plan(payload.get("plan"), difficulty)
        if is_valid:
            reviewed_plan = candidate_plan
            break
        last_reason = reason

    return {
        "routing": routing,
        "plan": reviewed_plan,
    }
