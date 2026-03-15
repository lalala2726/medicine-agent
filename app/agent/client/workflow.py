from __future__ import annotations

from typing import Any

from langgraph.constants import END, START
from langgraph.graph import StateGraph

from app.agent.admin.state import AgentState
from app.agent.client.after_sale_node import after_sale_agent
from app.agent.client.chat_node import chat_agent
from app.agent.client.gateway_node import gateway_router


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
    graph.add_node("after_sale_agent", after_sale_agent)

    graph.add_edge(START, "gateway_router")
    graph.add_conditional_edges(
        "gateway_router",
        _route_from_gateway,
        {
            "chat_agent": "chat_agent",
            "after_sale_agent": "after_sale_agent",
        },
    )
    graph.add_edge("chat_agent", END)
    graph.add_edge("after_sale_agent", END)

    return graph.compile()
