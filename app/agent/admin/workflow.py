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
        {
            "order_agent": "order_agent",
            "excel_agent": "excel_agent",
            "chart_agent": "chart_agent",
            END: END,
        },
    )
    return graph.compile()


def _flatten_plan(plan: list[PlanStep | list[PlanStep]] | None) -> list[PlanStep]:
    """将 plan 拉平为顺序步骤列表，先按顺序实现动态串行编排。"""
    if not plan:
        return []

    flattened: list[PlanStep] = []
    for item in plan:
        if isinstance(item, list):
            for step in item:
                if isinstance(step, dict):
                    flattened.append(step)
            continue
        if isinstance(item, dict):
            flattened.append(item)
    return flattened


def router(state: AgentState) -> dict[str, Any]:
    """
    根据 state['plan'] 动态计算下一跳节点。

    示例:
    - plan = [A, B, C] -> supervisor_agent -> A -> B -> C
    - plan = [A, C]    -> supervisor_agent -> A -> C
    """
    plan = _flatten_plan(state.get("plan"))
    routing = dict(state.get("routing") or {})
    plan_index = int(routing.get("plan_index", 0))

    next_node = END
    # 从当前游标开始找下一个有效节点，找到后游标前进 1
    while plan_index < len(plan):
        step = plan[plan_index]
        plan_index += 1
        candidate = step.get("node_name")
        if candidate in EXECUTION_NODES:
            next_node = candidate
            break

    routing["plan_index"] = plan_index
    routing["next_node"] = next_node
    return {"routing": routing}


def _route_to_next_node(state: AgentState) -> str:
    """读取 router 计算结果，返回下一跳节点名；没有可执行步骤则 END。"""
    next_node = (state.get("routing") or {}).get("next_node", END)
    if next_node in EXECUTION_NODES:
        return next_node
    return END
