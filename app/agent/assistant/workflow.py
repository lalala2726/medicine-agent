from __future__ import annotations

from typing import Any

from langgraph.constants import END, START
from langgraph.graph import StateGraph

from app.agent.assistant.node import (
    chat_agent,
    gateway_router,
    supervisor_agent,
)
from app.agent.assistant.state import AgentState


def _route_from_gateway(state: AgentState) -> str:
    routing = state.get("routing")
    if isinstance(routing, dict):
        route_target = str(routing.get("route_target") or "").strip()
        if route_target == "chat_agent":
            return "chat_agent"
        if route_target == "supervisor_agent":
            return "supervisor_agent"

    return "supervisor_agent"


def build_graph() -> Any:
    graph = StateGraph(AgentState)

    graph.add_node("gateway_router", gateway_router)
    graph.add_node("supervisor_agent", supervisor_agent)
    graph.add_node("chat_agent", chat_agent)

    graph.add_edge(START, "gateway_router")
    graph.add_conditional_edges(
        "gateway_router",
        _route_from_gateway,
        {
            "chat_agent": "chat_agent",
            "supervisor_agent": "supervisor_agent",
        },
    )
    graph.add_edge("chat_agent", END)
    graph.add_edge("supervisor_agent", END)

    return graph.compile()
