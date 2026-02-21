from __future__ import annotations

from typing import Any

from langgraph.constants import END, START
from langgraph.graph import StateGraph

from app.agent.admin.node import (
    chat_agent,
    gateway_router,
    supervisor_agent,
)
from app.agent.admin.state import AgentState


def build_graph() -> Any:
    graph = StateGraph(AgentState)

    graph.add_node("gateway_router", gateway_router)
    graph.add_node("supervisor_agent", supervisor_agent)
    graph.add_node("chat_agent", chat_agent)

    graph.add_edge(START, "gateway_router")
    graph.add_edge("chat_agent", END)

    return graph.compile()
