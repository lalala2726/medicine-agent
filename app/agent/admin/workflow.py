from __future__ import annotations

from typing import Any

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

# gateway 条件路由映射：决定入口分流到哪个节点。
GATEWAY_ROUTE_MAP = {
    "order_agent": "order_agent",
    "product_agent": "product_agent",
    "chat_agent": "chat_agent",
    "supervisor_agent": "supervisor_agent",
    END: END,
}

# supervisor 条件路由映射：决定慢车道下一跳或结束。
SUPERVISOR_ROUTE_MAP = {
    "order_agent": "order_agent",
    "product_agent": "product_agent",
    "excel_agent": "excel_agent",
    END: END,
}

# worker 执行后路由映射：快车道结束，慢车道强制回 supervisor。
WORKER_NEXT_MAP = {
    "supervisor_agent": "supervisor_agent",
    END: END,
}


def _route_from_gateway(state: AgentState) -> str:
    """
    读取 gateway 决策并返回下一跳节点名。

    Args:
        state: 当前图状态，需包含 `routing.route_target`。

    Returns:
        str: 合法路由目标；若目标非法则回退 `supervisor_agent`。
    """

    routing = state.get("routing") or {}
    route_target = str(routing.get("route_target") or "supervisor_agent")
    if route_target in {"order_agent", "product_agent", "chat_agent", "supervisor_agent"}:
        return route_target
    return "supervisor_agent"


def _route_from_supervisor(state: AgentState) -> str:
    """
    读取 supervisor 输出并决定下一跳。

    Args:
        state: 当前图状态，需包含 `next_node`。

    Returns:
        str: 当 `next_node == FINISH` 时返回 `END`，
            否则在允许集合中返回业务节点名；非法值统一回退 `END`。
    """

    next_node = str(state.get("next_node") or "FINISH")
    if next_node == "FINISH":
        return END
    if next_node in {"order_agent", "product_agent", "excel_agent"}:
        return next_node
    return END


def _route_after_worker(state: AgentState) -> str:
    """
    worker 节点执行后决定去向。

    Args:
        state: 当前图状态，主要读取 `routing.mode`。

    Returns:
        str: `fast_lane` 返回 `END`（直达结束）；
            其他模式返回 `supervisor_agent`（慢车道回环）。
    """

    routing = state.get("routing") or {}
    mode = str(routing.get("mode") or "")
    if mode == "fast_lane":
        return END
    return "supervisor_agent"


def build_graph() -> Any:
    """
    构建并编译 Gateway + Supervisor 多 Agent 工作流图。

    拓扑结构：
    1. `START -> gateway_router`；
    2. `gateway_router -> order_agent|product_agent|chat_agent|supervisor_agent`；
    3. `supervisor_agent -> order_agent|product_agent|excel_agent|END`；
    4. `order/product/excel -> END(快车道) 或 supervisor_agent(慢车道)`；
    5. `chat_agent -> END`。

    Args:
        无。

    Returns:
        Any: 可直接执行的 LangGraph 编译对象。
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
