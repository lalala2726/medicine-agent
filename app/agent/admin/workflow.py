from __future__ import annotations

from typing import Any

from langgraph.constants import END, START
from langgraph.graph import StateGraph

from agent.admin.coordinator_node import summary_agent
from app.agent.admin.agent_state import AgentState, PlanStep
from app.agent.admin.chart_node import chart_agent
from app.agent.admin.excel_node import excel_agent
from app.agent.admin.order_node import order_agent
from app.agent.admin.supervisor_node import coordinator

# 当前图中允许被路由到的业务节点
EXECUTION_NODES = ("order_agent", "excel_agent", "chart_agent")


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("chart_agent", chart_agent)
    graph.add_node("excel_agent", excel_agent)
    graph.add_node("order_agent", order_agent)
    graph.add_node("summery_agent", summary_agent)
    graph.add_node("coordinator_agent", coordinator)
    graph.add_node("planner", planner)
    graph.add_node("router", gateway_router)

    graph.add_edge(START, "gateway_router")
    graph.add_edge("gateway_router", "router")
    graph.add_conditional_edges(

    )

    # 业务节点执行完后都回到 router，继续按 plan 选择下一步
    for node in EXECUTION_NODES:
        graph.add_edge(node, "planner")

    graph.add_conditional_edges(
        "planner",
        _route_to_next_node,
        {**{node: node for node in EXECUTION_NODES}, END: END},
    )
    return graph.compile()


def _normalize_plan_stages(plan: list[PlanStep | list[PlanStep]] | None) -> list[list[PlanStep]]:
    """将 plan 规范化为阶段列表: 单节点=串行阶段, 列表=并行阶段。"""
    if not plan:
        return []

    stages: list[list[PlanStep]] = []
    for item in plan:
        if isinstance(item, list):
            stage = []
            for step in item:
                if isinstance(step, dict):
                    stage.append(step)
        elif isinstance(item, dict):
            stage = [item]
        else:
            stage = []

        if stage:
            stages.append(stage)

    return stages



def gateway_router(state: AgentState) -> str:
    system_prompt = """
        您是药品商城后台的 AI 管理助手，负责对用户请求进行统一决策
        你的核心职责包括
        1，根据用户的意图，如果是单一任务的话进行输出专门的字段
        如果是多个任务的话这边路由到专门的协调节点，并且根据难度分为简单任务，复杂任务
        复杂任务的话你还需要指定更加强大的模型来执行，并且这个切换模型只针对协调节点
    """
    pass


def planner(state: AgentState) -> dict[str, Any]:
    """
    根据 state['plan'] 动态计算下一跳节点（支持并行阶段）。

    示例:
    - plan = [A, B, C]     -> coordinator_agent -> A -> B -> C
    - plan = [A, C]        -> coordinator_agent -> A -> C
    - plan = [[A, B], C]   -> coordinator_agent -> (A || B) -> C
    """
    stages = _normalize_plan_stages(state.get("plan"))
    routing = dict(state.get("routing") or {})
    stage_index = int(routing.get("stage_index", 0))

    next_nodes: list[str] = []
    # 从当前游标开始找下一个有效阶段，找到后游标前进 1
    while stage_index < len(stages):
        stage = stages[stage_index]
        stage_index += 1

        candidate_nodes: list[str] = []
        for step in stage:
            candidate = step.get("node_name")
            if candidate in EXECUTION_NODES and candidate not in candidate_nodes:
                candidate_nodes.append(candidate)

        if candidate_nodes:
            next_nodes = candidate_nodes
            break

    routing["stage_index"] = stage_index
    routing["next_nodes"] = next_nodes
    return {"routing": routing}


def _route_to_next_node(state: AgentState) -> str | list[str]:
    """读取 router 计算结果；单节点返回 str，并行节点返回 list[str]。"""
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
