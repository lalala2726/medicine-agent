from __future__ import annotations

from typing import Any

from langgraph.constants import END, START
from langgraph.graph import StateGraph

from app.agent.client.domain.after_sale import after_sale_agent
from app.agent.client.domain.chat import chat_agent
from app.agent.client.domain.consultation import consultation_agent
from app.agent.client.domain.order import order_agent
from app.agent.client.domain.product import product_agent
from app.agent.client.domain.router import gateway_router
from app.agent.client.state import AgentState


def _route_from_gateway(state: AgentState) -> str:
    """根据 client gateway 结果选择下一个执行节点。"""

    routing = state.get("routing")
    if isinstance(routing, dict):
        raw_targets = routing.get("route_targets")
        if not isinstance(raw_targets, list):
            return "chat_agent"

        normalized_targets: list[str] = []
        allowed_targets = {
            "chat_agent",
            "consultation_agent",
            "order_agent",
            "product_agent",
            "after_sale_agent",
        }
        for raw_target in raw_targets:
            target = str(raw_target or "").strip()
            if not target:
                return "chat_agent"
            if target not in allowed_targets:
                return "chat_agent"
            if target in normalized_targets:
                continue
            normalized_targets.append(target)

        if len(normalized_targets) != 1:
            return "chat_agent"
        return normalized_targets[0]

    return "chat_agent"


def build_graph() -> Any:
    """构建 client assistant graph。"""

    graph = StateGraph(AgentState)

    graph.add_node("gateway_router", gateway_router)
    graph.add_node("chat_agent", chat_agent)
    graph.add_node("consultation_agent", consultation_agent)
    graph.add_node("order_agent", order_agent)
    graph.add_node("product_agent", product_agent)
    graph.add_node("after_sale_agent", after_sale_agent)

    graph.add_edge(START, "gateway_router")
    graph.add_conditional_edges(
        "gateway_router",
        _route_from_gateway,
        {
            "chat_agent": "chat_agent",
            "consultation_agent": "consultation_agent",
            "order_agent": "order_agent",
            "product_agent": "product_agent",
            "after_sale_agent": "after_sale_agent",
        },
    )
    graph.add_edge("chat_agent", END)
    graph.add_edge("consultation_agent", END)
    graph.add_edge("order_agent", END)
    graph.add_edge("product_agent", END)
    graph.add_edge("after_sale_agent", END)

    return graph.compile()
