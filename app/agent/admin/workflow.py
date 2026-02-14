from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.constants import END, START
from langgraph.graph import StateGraph

from app.agent.admin.agent_state import AgentState, PlanStep
from app.agent.admin.node import (
    chat_agent,
    chart_agent,
    coordinator,
    excel_agent,
    order_agent,
    summary_agent,
)
from app.agent.admin.node.product_node import product_agent
from app.core.langsmith import traceable
from app.core.llm import create_chat_model

# 当前图中允许被执行的业务节点
EXECUTION_NODES = (
    "order_agent",
    "excel_agent",
    "chart_agent",
    "summary_agent",
    "product_agent",
)
GATEWAY_ROUTE_NODES = (*EXECUTION_NODES, "chat_agent", "coordinator_agent")
GATEWAY_ROUTE_MAP = {
    "order_agent": "order_agent",
    "excel_agent": "excel_agent",
    "chart_agent": "chart_agent",
    "summary_agent": "summary_agent",
    "product_agent": "product_agent",
    "chat_agent": "chat_agent",
    "coordinator_agent": "coordinator_agent",
    END: END,
}
PLANNER_ROUTE_MAP = {
    "order_agent": "order_agent",
    "excel_agent": "excel_agent",
    "chart_agent": "chart_agent",
    "summary_agent": "summary_agent",
    "product_agent": "product_agent",
    END: END,
}

GATEWAY_ROUTER_PROMPT = """
    你是药品商城后台的网关路由节点（gateway_router）。
    你的任务是根据用户请求，决定是直接交给单个业务节点，还是交给 coordinator_agent 协调多节点执行。

    可用节点与业务范围：
    1. order_agent
       - 订单域任务：订单查询、订单状态判断、订单信息核验、订单明细检索。
    2. excel_agent
       - 表格域任务：支持将系统数据按照条件导出为excel表格并生成下载链接让用户下载
       - 支持如下:用户数据导出、订单数据导出、商品数据导出
       - 如果用户想要导出的数据不在以上范围内，你应该直接调用 chat_agent 并且告诉用户原因
       - 请注意！此Agent不接受导出数据范围以外的请求，并且这个agent有自己获取数据的能力不要为了数据调用其他的Agent
    3. chart_agent
       - 图表域任务：根据已有结构化数据生成图表、统计可视化结果说明。
    4. summary_agent
       - 汇总域任务：汇总多个节点结果，输出最终结论。
    5. chat_agent
        - 普通对话：非业务相关问题、咨询、闲聊。
    7. coordinator_agent
       - 协调域任务：多节点任务拆解、并行/串行编排、跨节点结果整合。
       - 当请求涉及两个及以上业务域，或依赖关系复杂时，必须路由到该节点。
    8. product_agent
       - 商品域任务：商品信息查询、商品信息核验

    请严格输出 JSON 对象，且只输出 JSON，不要包含其他文字：
    {
      "route_target": "order_agent | excel_agent | chart_agent | summary_agent | chat_agent | coordinator_agent | product_agent",
      "difficulty": "simple | medium | complex"
    }

    路由规则：
    1. 如果单个业务节点即可完成，且不需要跨节点协作：
       - route_target 设置为对应业务节点
       - difficulty = simple
    2. 如果任务涉及多个业务域、需要拆解并行/串行步骤、或存在明显依赖关系：
       - route_target = coordinator_agent
       - difficulty 按复杂度设置为 medium 或 complex
    3. 如果意图不清晰，默认 route_target = chat_agent, difficulty = simple。
"""


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("gateway_router", gateway_router)
    graph.add_node("coordinator_agent", coordinator)
    graph.add_node("planner", planner)
    graph.add_node("chat_agent", chat_agent)
    graph.add_node("summary_agent", summary_agent)
    graph.add_node("order_agent", order_agent)
    graph.add_node("excel_agent", excel_agent)
    graph.add_node("chart_agent", chart_agent)
    graph.add_node("product_agent", product_agent)

    graph.add_edge(START, "gateway_router")
    graph.add_conditional_edges(
        "gateway_router",
        _route_from_gateway,
        GATEWAY_ROUTE_MAP,
    )

    graph.add_edge("coordinator_agent", "planner")
    graph.add_edge("chat_agent", END)

    for node in EXECUTION_NODES:
        graph.add_edge(node, "planner")

    graph.add_conditional_edges(
        "planner",
        _route_to_next_node,
        PLANNER_ROUTE_MAP,
    )
    return graph.compile()


def _normalize_plan_steps(plan: list[PlanStep] | None) -> list[PlanStep]:
    # planner 运行前再做一次保护性清洗：没有 step_id 的步骤直接忽略。
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


def _normalize_difficulty(value: str) -> str:
    difficulty = (value or "simple").strip().lower()
    if difficulty in {"simple", "medium", "complex"}:
        return difficulty
    return "medium"


def _extract_status(step_outputs: dict[str, Any]) -> dict[str, str]:
    # 从节点写回的 step_outputs 中提取调度所需状态。
    # planner 只关心 completed/failed/skipped 三种终态。
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
    # 下游被依赖失败阻断时，由 planner 统一写 skipped，
    # 这样最终输出能解释“为什么没执行”。
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


@traceable(name="Gateway Router Node", run_type="chain")
def gateway_router(state: AgentState) -> dict[str, Any]:
    """
    第一跳网关节点：判断直接执行还是进入 coordinator_agent。
    """
    user_input = str(state.get("user_input") or "").strip()
    routing = dict(state.get("routing") or {})

    route_target = "coordinator_agent"
    difficulty = "medium"
    if user_input:
        try:
            model = create_chat_model(
                model="qwen-flash",
                max_tokens=1024,
                temperature=0,
                response_format={"type": "json_object"},
            )
            messages = [
                SystemMessage(content=GATEWAY_ROUTER_PROMPT),
                HumanMessage(content=f"用户请求：{user_input}"),
            ]
            response = model.invoke(messages)
            raw = json.loads(str(response.content))
            route_target = str(raw["route_target"])
            difficulty = str(raw["difficulty"])
        except Exception:
            route_target = "coordinator_agent"
            difficulty = "medium"

    if route_target not in GATEWAY_ROUTE_NODES:
        route_target = "coordinator_agent"

    # 初始化 DAG 调度相关运行态字段，保证后续节点读状态时不判空。
    routing["route_target"] = route_target
    routing["difficulty"] = _normalize_difficulty(difficulty)
    routing.setdefault("stage_index", 0)
    routing.setdefault("next_nodes", [])
    routing.setdefault("next_step_ids", [])
    routing.setdefault("current_step_ids", [])
    routing.setdefault("completed_step_ids", [])
    routing.setdefault("blocked_step_ids", [])

    return {"routing": routing}


def _route_from_gateway(state: AgentState) -> str:
    routing = state.get("routing") or {}
    route_target = routing.get("route_target")
    if route_target in GATEWAY_ROUTE_NODES:
        return route_target
    return END


@traceable(name="Planner Node", run_type="chain")
def planner(state: AgentState) -> dict[str, Any]:
    """
    DAG 调度器：
    - 基于 depends_on 判定 ready 步骤
    - 上游失败时阻断并写 skipped
    - 并行时同 node_name 仅放行一个
    """
    # 1) 读取并清洗计划
    plan_steps = _normalize_plan_steps(state.get("plan"))
    routing = dict(state.get("routing") or {})

    if not plan_steps:
        routing["next_nodes"] = []
        routing["next_step_ids"] = []
        routing["current_step_ids"] = []
        routing["current_step_map"] = {}
        routing["is_final_stage"] = False
        return {"routing": routing}

    # 2) 建立 step_id -> step 映射，并保留原始顺序（用于稳定调度和“同节点串行化”）。
    step_by_id: dict[str, PlanStep] = {}
    ordered_step_ids: list[str] = []
    for step in plan_steps:
        step_id = str(step.get("step_id") or "").strip()
        if step_id and step_id not in step_by_id:
            step_by_id[step_id] = step
            ordered_step_ids.append(step_id)

    # 3) 基于 step_outputs 识别已完成/失败/跳过的终态步骤。
    raw_step_outputs = state.get("step_outputs") or {}
    step_outputs = raw_step_outputs if isinstance(raw_step_outputs, dict) else {}
    status_by_id = _extract_status(step_outputs)

    completed_ids = {sid for sid, status in status_by_id.items() if status == "completed"}
    failed_or_skipped_ids = {
        sid for sid, status in status_by_id.items() if status in {"failed", "skipped"}
    }
    terminal_ids = set(status_by_id.keys())

    # 4) 传播式阻断：
    # 如果某步骤依赖 failed/skipped 步骤，则该步骤标记 blocked（待写 skipped）。
    blocked_ids: set[str] = set()
    blocked_reasons: dict[str, list[str]] = {}
    changed = True
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

    # 5) 为新阻断的步骤生成 skipped 输出（仅写一次）。
    blocked_updates: dict[str, Any] = {}
    for blocked_step_id in blocked_ids:
        if blocked_step_id in status_by_id:
            continue
        step = step_by_id.get(blocked_step_id) or {}
        blocked_updates[blocked_step_id] = _build_skipped_step_output(
            step,
            failed_dependencies=blocked_reasons.get(blocked_step_id, []),
        )

    # 6) 选取 ready 步骤：
    # - 依赖全部 completed
    # - 尚未终态/未阻断
    # - 同一轮同 node_name 仅放行一个（避免图节点重复并发）
    ready_steps: list[PlanStep] = []
    used_nodes: set[str] = set()
    terminal_or_blocked = terminal_ids | blocked_ids
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
    current_step_map = {
        str(step.get("node_name")): step for step in ready_steps if step.get("node_name")
    }

    # 7) 回写 routing 运行态，供节点和 SSE 层判断流程状态。
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
    # final_output 语义：当前轮若包含 final_output=true 的步骤，则该步骤视为最终输出节点。
    routing["is_final_stage"] = any(bool(step.get("final_output")) for step in ready_steps)
    try:
        stage_index = int(routing.get("stage_index", 0))
    except (TypeError, ValueError):
        stage_index = 0
    routing["stage_index"] = stage_index + 1

    # 8) 返回 routing + 可选的 blocked step_outputs 增量。
    result: dict[str, Any] = {"routing": routing}
    if blocked_updates:
        result["step_outputs"] = blocked_updates
    return result


def _route_to_next_node(state: AgentState) -> str | list[str]:
    routing = state.get("routing") or {}
    raw_next_nodes = routing.get("next_nodes")
    if not isinstance(raw_next_nodes, list):
        raw_next_nodes = []

    next_nodes = [node for node in raw_next_nodes if node in EXECUTION_NODES]
    if not next_nodes:
        return END
    if len(next_nodes) == 1:
        return next_nodes[0]
    return next_nodes
