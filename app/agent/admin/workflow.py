from __future__ import annotations

from typing import Any

from langgraph.constants import END, START
from langgraph.graph import StateGraph

from app.agent.admin.agent_state import AgentState, PlanStep
from app.agent.admin.chart_agent import chart_agent
from app.agent.admin.excel_agent import excel_agent
from app.agent.admin.order_agent import order_agent
from app.agent.admin.supervisor_agent import supervisor_agent

# 当前图中允许被路由到的业务节点
EXECUTION_NODES = ("order_agent", "excel_agent", "chart_agent")


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("chart_agent", chart_agent)
    graph.add_node("excel_agent", excel_agent)
    graph.add_node("order_agent", order_agent)
    graph.add_node("supervisor_agent", supervisor_agent)
    graph.add_node("router", router)

    graph.add_edge(START, "supervisor_agent")
    graph.add_edge("supervisor_agent", "router")

    # 业务节点执行完后都回到 router，继续按 plan 选择下一步
    for node in EXECUTION_NODES:
        graph.add_edge(node, "router")

    graph.add_conditional_edges(
        "router",
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
            stage = [step for step in item if isinstance(step, dict)]
        elif isinstance(item, dict):
            stage = [item]
        else:
            stage = []

        if stage:
            stages.append(stage)

    return stages


def router(state: AgentState) -> dict[str, Any]:
    """
    根据 state['plan'] 动态计算下一跳节点（支持并行阶段）。

    示例:
    - plan = [A, B, C]     -> supervisor_agent -> A -> B -> C
    - plan = [A, C]        -> supervisor_agent -> A -> C
    - plan = [[A, B], C]   -> supervisor_agent -> (A || B) -> C
    """
    stages = _normalize_plan_stages(state.get("plan"))
    routing = dict(state.get("routing") or {})
    # 兼容旧字段 plan_index，优先使用新字段 stage_index
    stage_index = int(routing.get("stage_index", routing.get("plan_index", 0)))

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
    # 保留旧字段，避免外部依赖立即失效
    routing["plan_index"] = stage_index
    routing["next_node"] = next_nodes[0] if len(next_nodes) == 1 else END
    return {"routing": routing}


def _route_to_next_node(state: AgentState) -> str | list[str]:
    """读取 router 计算结果；单节点返回 str，并行节点返回 list[str]。"""
    routing = state.get("routing") or {}
    raw_next_nodes = routing.get("next_nodes")

    if not isinstance(raw_next_nodes, list):
        raw_next_nodes = [routing.get("next_node")]

    next_nodes = [node for node in raw_next_nodes if node in EXECUTION_NODES]
    if not next_nodes:
        return END
    if len(next_nodes) == 1:
        return next_nodes[0]
    return next_nodes
