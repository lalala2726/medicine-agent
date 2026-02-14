import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.admin.agent_state import AgentState
from app.core.assistant_status import status_node
from app.core.langsmith import traceable
from app.core.llm import create_chat_model

_system_prompt = """
你是后台工作流中的 coordinator_agent，负责把用户请求拆解成可执行计划。

你的职责：
1. 根据任务依赖关系生成 plan（支持串行与并行阶段）。
2. 只做规划，不执行具体业务。

可用执行节点（node_name 只能从以下值中选择）：
- order_agent: 订单查询、订单状态与订单信息核验。
- excel_agent: 表格解析、表格整理、Excel 导出。
- chart_agent: 基于已有结构化数据生成图表或统计说明。
- summary_agent: 对多个节点结果做汇总并输出最终结论。
- product_agent: 商品查询、商品详情查询

注意：
- chat_agent 由 gateway_router 处理，不要出现在 plan 中。

你输出的 JSON 必须与状态结构兼容（对应 AgentState 中的 plan）：
{
  "plan": [
    {
      "node_name": "order_agent",
      "task_description": "string",
      "last_node": ["coordinator_agent"]
    },
    [
      {
        "node_name": "excel_agent",
        "task_description": "string",
        "last_node": ["order_agent"]
      },
      {
        "node_name": "chart_agent",
        "task_description": "string",
        "last_node": ["order_agent"]
      }
    ]
  ]
}

严格规则：
1. 只输出 JSON，不要输出任何解释、Markdown、代码块标记。
2. 顶层只允许一个键：plan。
3. plan 必须是数组。
4. plan 每个元素要么是单个步骤对象（串行），要么是步骤对象数组（并行阶段）。
5. 每个步骤对象必须包含：node_name、task_description、last_node。
7. last_node 必须是字符串数组：
   - 第一阶段通常为 ["coordinator_agent"]。
   - 后续阶段填写其依赖的上游节点名数组。
8. 不要产生循环依赖，不要把 coordinator_agent 作为 plan 的 node_name。
9. 若用户需求不清晰，给出最小可执行 plan（至少 1 个步骤）。
"""

_COORDINATOR_MODEL_BY_DIFFICULTY = {
    "simple": "qwen-flash",
    "medium": "qwen-plus",
    "complex": "qwen-max",
}

_PLAN_ALLOWED_NODES = {"order_agent", "excel_agent", "chart_agent", "summary_agent", "product_agent"}
_MAX_PLAN_STEPS_BY_DIFFICULTY = {
    "simple": 1,
    "medium": 3,
    "complex": 6,
}
_PLAN_REVIEW_RETRY_LIMIT = 2


def _select_model_by_difficulty(difficulty: str) -> str:
    key = str(difficulty or "simple").strip().lower()
    return _COORDINATOR_MODEL_BY_DIFFICULTY.get(key, _COORDINATOR_MODEL_BY_DIFFICULTY["simple"])


def _difficulty_step_limit(difficulty: str) -> int:
    key = str(difficulty or "medium").strip().lower()
    return _MAX_PLAN_STEPS_BY_DIFFICULTY.get(key, _MAX_PLAN_STEPS_BY_DIFFICULTY["medium"])


def _normalize_plan(plan: Any) -> list[list[dict[str, Any]]]:
    """将模型返回的 plan 统一为阶段列表，便于后续执行统一校验。"""
    if not isinstance(plan, list):
        return []

    stages: list[list[dict[str, Any]]] = []
    for item in plan:
        if isinstance(item, dict):
            stage = [item]
        elif isinstance(item, list):
            stage = [step for step in item if isinstance(step, dict)]
        else:
            stage = []
        if stage:
            stages.append(stage)
    return stages


def review_plan(plan: Any, difficulty: str) -> tuple[bool, list[dict | list[dict]], str]:
    """
    审核协调器生成的 plan，防止非法节点与过高复杂度计划进入执行环节。

    校验策略：
    1. 防循环依赖：严禁 node_name = coordinator_agent。
    2. 防越界：node_name 只能是当前已实现的业务节点。
    3. 控复杂度：总步骤数不超过当前 difficulty 的上限。
    4. 保结构完整：每个步骤必须有 node_name / task_description / last_node。

    Returns:
        (is_valid, normalized_plan, reason)
        - is_valid: 是否通过校验
        - normalized_plan: 规范化后的计划（可直接写回 state.plan）
        - reason: 失败原因（用于反馈模型重生）
    """
    stages = _normalize_plan(plan)
    if not stages:
        return False, [], "plan 为空或结构非法，至少需要一个可执行步骤。"

    total_steps = 0
    max_steps = _difficulty_step_limit(difficulty)
    normalized_plan: list[dict | list[dict]] = []

    for stage_index, stage in enumerate(stages, start=1):
        if len(stage) > len(_PLAN_ALLOWED_NODES):
            return False, [], f"第{stage_index}阶段并行节点过多，超过当前已实现节点数量。"

        stage_nodes: set[str] = set()
        normalized_stage: list[dict[str, Any]] = []

        for step in stage:
            node_name = str(step.get("node_name") or "").strip()
            if not node_name:
                return False, [], f"第{stage_index}阶段存在缺失 node_name 的步骤。"
            if node_name == "coordinator_agent":
                return False, [], "plan 中出现 coordinator_agent，会造成循环依赖。"
            if node_name not in _PLAN_ALLOWED_NODES:
                return False, [], f"节点 {node_name} 未实现，不可出现在 plan 中。"
            if node_name in stage_nodes:
                return False, [], f"第{stage_index}阶段出现重复节点 {node_name}。"
            stage_nodes.add(node_name)

            task_description = step.get("task_description")
            if not isinstance(task_description, str) or not task_description.strip():
                return False, [], f"节点 {node_name} 缺少有效 task_description。"

            raw_last_node = step.get("last_node")
            if isinstance(raw_last_node, list):
                last_node = [str(item) for item in raw_last_node if str(item)]
            else:
                last_node = ["coordinator_agent"] if stage_index == 1 else []

            normalized_stage.append(
                {
                    "node_name": node_name,
                    "task_description": task_description.strip(),
                    "last_node": last_node,
                }
            )

            total_steps += 1
            if total_steps > max_steps:
                return False, [], f"plan 复杂度过高（步骤数 {total_steps} 超过 {difficulty} 允许上限 {max_steps}）。"

        if len(normalized_stage) == 1:
            normalized_plan.append(normalized_stage[0])
        else:
            normalized_plan.append(normalized_stage)

    return True, normalized_plan, "ok"


def _build_retry_feedback(reason: str) -> str:
    return (
        "上一次生成的 plan 未通过系统校验，原因："
        f"{reason}。\n"
        "请重新生成完整 JSON（仅包含 plan），并严格遵守："
        "不要把 coordinator_agent 作为 plan 的 node_name，避免循环依赖。"
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
    reviewed_plan: list[dict | list[dict]] = []
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

        is_valid, candidate_plan, reason = review_plan(payload.get("plan"), difficulty)
        if is_valid:
            reviewed_plan = candidate_plan
            break
        last_reason = reason

    return {
        "routing": routing,
        "plan": reviewed_plan,
    }
