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
from app.core.assistant_status import status_node
from app.core.langsmith import traceable
from app.core.llm import create_chat_model

# 当前图中允许被执行的业务节点
EXECUTION_NODES = ("order_agent", "excel_agent", "chart_agent", "summary_agent")
GATEWAY_ROUTE_NODES = (*EXECUTION_NODES, "chat_agent", "coordinator_agent")
GATEWAY_ROUTE_MAP = {
    "order_agent": "order_agent",
    "excel_agent": "excel_agent",
    "chart_agent": "chart_agent",
    "summary_agent": "summary_agent",
    "chat_agent": "chat_agent",
    "coordinator_agent": "coordinator_agent",
    END: END,
}
PLANNER_ROUTE_MAP = {
    "order_agent": "order_agent",
    "excel_agent": "excel_agent",
    "chart_agent": "chart_agent",
    "summary_agent": "summary_agent",
    END: END,
}

GATEWAY_ROUTER_PROMPT = """
    你是药品商城后台的网关路由节点（gateway_router）。
    你的任务是根据用户请求，决定是直接交给单个业务节点，还是交给 coordinator_agent 协调多节点执行。
    
    可用节点与业务范围：
    1. order_agent
       - 订单域任务：订单查询、订单状态判断、订单信息核验、订单明细检索。
    2. excel_agent
       - 表格域任务：支持将结构化数据转换为 Excel 文件、数据整理、表格生成。
       - 此Agent有导出系统中的列表为表格的能力并且能进行复杂的公式计算。
       - 支持导出系统中数据为 Excel 文件如下：商品列表、订单列表、用户列表
    3. chart_agent
       - 图表域任务：根据已有结构化数据生成图表、统计可视化结果说明。
    4. summary_agent
       - 汇总域任务：汇总多个节点结果，输出最终结论。
    5. chat_agent
        - 普通对话：非业务相关问题、咨询、闲聊。
    7. coordinator_agent
       - 协调域任务：多节点任务拆解、并行/串行编排、跨节点结果整合。
       - 当请求涉及两个及以上业务域，或依赖关系复杂时，必须路由到该节点。
    
    请严格输出 JSON 对象，且只输出 JSON，不要包含其他文字：
    {
      "route_target": "order_agent | excel_agent | chart_agent | summary_agent | chat_agent | coordinator_agent",
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

    graph.add_edge(START, "gateway_router")
    graph.add_conditional_edges(
        "gateway_router",
        _route_from_gateway,
        GATEWAY_ROUTE_MAP,
    )

    # 进入协调器后，由 planner 按 plan 与阶段继续调度
    graph.add_edge("coordinator_agent", "planner")
    graph.add_edge("chat_agent", END)

    # 业务节点执行结束后回到 planner 决定下一步或结束
    for node in EXECUTION_NODES:
        graph.add_edge(node, "planner")

    graph.add_conditional_edges(
        "planner",
        _route_to_next_node,
        PLANNER_ROUTE_MAP,
    )
    return graph.compile()


def _normalize_plan_stages(plan: list[PlanStep | list[PlanStep]] | None) -> list[list[PlanStep]]:
    """将 plan 规范化为阶段列表: 单节点=串行阶段, 列表=并行阶段。"""
    if not plan:
        return []

    stages: list[list[PlanStep]] = []
    for item in plan:
        if isinstance(item, list):
            stage = [step for step in item if isinstance(step, dict)]
        elif isinstance(item, dict):
            stage = [item]
        else:
            stage = []

        if stage:
            stages.append(stage)

    return stages


def _normalize_difficulty(value: str) -> str:
    difficulty = (value or "simple").strip().lower()
    if difficulty in {"simple", "medium", "complex"}:
        return difficulty
    return "medium"


def _stage_has_executable_node(stage: list[PlanStep]) -> bool:
    for step in stage:
        if step.get("node_name") in EXECUTION_NODES:
            return True
    return False


@status_node(node="router", start_message="正在分析问题")
@traceable(name="Gateway Router Node", run_type="chain")
def gateway_router(state: AgentState) -> dict[str, Any]:
    """
    第一跳网关节点：判断直接执行还是进入 coordinator_agent。

    注意：
    - 本节点仅做路由决策，不做计划拆解。
    - 失败时默认走 coordinator_agent，保证流程可继续。
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

    routing["route_target"] = route_target
    routing["difficulty"] = _normalize_difficulty(difficulty)
    routing.setdefault("stage_index", 0)
    routing.setdefault("next_nodes", [])

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
    planner 根据 plan 的阶段返回 next_nodes（支持并行阶段）。
    """
    stages = _normalize_plan_stages(state.get("plan"))
    routing = dict(state.get("routing") or {})

    try:
        stage_index = int(routing.get("stage_index", 0))
    except (TypeError, ValueError):
        stage_index = 0

    next_nodes: list[str] = []
    current_step_map: dict[str, PlanStep] = {}
    # 从当前游标开始找下一个有效阶段，找到后游标前进 1
    while stage_index < len(stages):
        stage = stages[stage_index]
        stage_index += 1

        candidate_nodes: list[str] = []
        candidate_steps: dict[str, PlanStep] = {}
        for step in stage:
            candidate = step.get("node_name")
            if candidate in EXECUTION_NODES and candidate not in candidate_nodes:
                candidate_nodes.append(candidate)
                candidate_steps[candidate] = step

        if candidate_nodes:
            next_nodes = candidate_nodes
            current_step_map = candidate_steps
            break

    # 记录当前阶段是否为最后一个有效阶段，供节点决定是否走收尾流式输出。
    is_final_stage = True
    if next_nodes:
        for stage in stages[stage_index:]:
            if _stage_has_executable_node(stage):
                is_final_stage = False
                break

    routing["stage_index"] = stage_index
    routing["next_nodes"] = next_nodes
    routing["current_step_map"] = current_step_map
    routing["is_final_stage"] = is_final_stage
    return {"routing": routing}


def _route_to_next_node(state: AgentState) -> str | list[str]:
    """读取 planner 计算结果；单节点返回 str，并行节点返回 list[str]。"""
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
