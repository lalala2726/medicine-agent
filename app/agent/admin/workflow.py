from __future__ import annotations

from langgraph.constants import END, START
from langgraph.graph import StateGraph

from app.agent.admin.node import (
    chat_agent,
    excel_agent,
    gateway_router,
    order_agent,
    product_agent,
    supervisor_agent,
)
from app.agent.admin.state import AgentState

GATEWAY_ROUTE_MAP = {
    "order_agent": "order_agent",
    "product_agent": "product_agent",
    "chat_agent": "chat_agent",
    "supervisor_agent": "supervisor_agent",
    END: END,
}

SUPERVISOR_ROUTE_MAP = {
    "order_agent": "order_agent",
    "product_agent": "product_agent",
    "excel_agent": "excel_agent",
    END: END,
}

WORKER_NEXT_MAP = {
    "supervisor_agent": "supervisor_agent",
    END: END,
}


def _route_from_gateway(state: AgentState) -> str:
    routing = state.get("routing") or {}
    route_target = str(routing.get("route_target") or "supervisor_agent")
    if route_target in {"order_agent", "product_agent", "chat_agent", "supervisor_agent"}:
        return route_target
    return "supervisor_agent"


def _route_from_supervisor(state: AgentState) -> str:
    next_node = str(state.get("next_node") or "FINISH")
    if next_node == "FINISH":
        return END
    if next_node in {"order_agent", "product_agent", "excel_agent"}:
        return next_node
    return END


def _route_after_worker(state: AgentState) -> str:
    routing = state.get("routing") or {}
    mode = str(routing.get("mode") or "")
    if mode == "fast_lane":
        return END
    return "supervisor_agent"


def build_graph():
    """
    Build Gateway + Supervisor graph.
    """
    graph = StateGraph(AgentState)

    graph.add_node("gateway_router", gateway_router)
    graph.add_node("supervisor_agent", supervisor_agent)
    graph.add_node("order_agent", order_agent)
    graph.add_node("product_agent", product_agent)
    graph.add_node("excel_agent", excel_agent)
    graph.add_node("chat_agent", chat_agent)

    graph.add_edge(START, "gateway_router")
    graph.add_conditional_edges(
        "gateway_router",
        _route_from_gateway,
        GATEWAY_ROUTE_MAP,
    )

    graph.add_conditional_edges(
        "supervisor_agent",
        _route_from_supervisor,
        SUPERVISOR_ROUTE_MAP,
    )

    graph.add_conditional_edges("order_agent", _route_after_worker, WORKER_NEXT_MAP)
    graph.add_conditional_edges("product_agent", _route_after_worker, WORKER_NEXT_MAP)
    graph.add_conditional_edges("excel_agent", _route_after_worker, WORKER_NEXT_MAP)
    graph.add_edge("chat_agent", END)

    return graph.compile()

