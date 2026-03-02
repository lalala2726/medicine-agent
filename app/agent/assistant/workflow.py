from __future__ import annotations

from typing import Any

from langgraph.constants import END, START
from langgraph.graph import StateGraph

from app.agent.assistant.node import (
    adaptive_agent,
    after_sale_agent,
    analytics_agent,
    chat_agent,
    gateway_router,
    order_agent,
    product_agent,
    user_agent,
)
from app.agent.assistant.state import AgentState


def _route_from_gateway(state: AgentState) -> str:
    """
    功能描述：
        根据 gateway 输出的 `route_targets` 数组选择下一个执行节点。

    参数说明：
        state (AgentState): 工作流状态，期望包含 `routing.route_targets`。

    返回值：
        str:
            下一个节点名称，规则如下：
            - 单目标且为 `chat_agent`：返回 `chat_agent`；
            - 单目标且为业务域节点：返回该业务域节点；
            - 目标数量 >= 2：返回 `adaptive_agent`；
            - 缺失或非法：返回 `chat_agent`。

    异常说明：
        无；异常输入统一走 `chat_agent` 兜底。
    """

    routing = state.get("routing")
    if isinstance(routing, dict):
        raw_targets = routing.get("route_targets")
        if not isinstance(raw_targets, list):
            return "chat_agent"

        normalized_targets: list[str] = []
        allowed_targets = {
            "chat_agent",
            "order_agent",
            "product_agent",
            "after_sale_agent",
            "user_agent",
            "analytics_agent",
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

        if not normalized_targets:
            return "chat_agent"
        if "chat_agent" in normalized_targets and len(normalized_targets) > 1:
            return "chat_agent"
        if len(normalized_targets) >= 2:
            return "adaptive_agent"
        return normalized_targets[0]

    return "chat_agent"


def build_graph() -> Any:
    graph = StateGraph(AgentState)

    graph.add_node("gateway_router", gateway_router)
    graph.add_node("chat_agent", chat_agent)
    graph.add_node("order_agent", order_agent)
    graph.add_node("product_agent", product_agent)
    graph.add_node("after_sale_agent", after_sale_agent)
    graph.add_node("user_agent", user_agent)
    graph.add_node("analytics_agent", analytics_agent)
    graph.add_node("adaptive_agent", adaptive_agent)

    graph.add_edge(START, "gateway_router")
    graph.add_conditional_edges(
        "gateway_router",
        _route_from_gateway,
        {
            "chat_agent": "chat_agent",
            "order_agent": "order_agent",
            "product_agent": "product_agent",
            "after_sale_agent": "after_sale_agent",
            "user_agent": "user_agent",
            "analytics_agent": "analytics_agent",
            "adaptive_agent": "adaptive_agent",
        },
    )
    graph.add_edge("chat_agent", END)
    graph.add_edge("order_agent", END)
    graph.add_edge("product_agent", END)
    graph.add_edge("after_sale_agent", END)
    graph.add_edge("user_agent", END)
    graph.add_edge("analytics_agent", END)
    graph.add_edge("adaptive_agent", END)

    return graph.compile()
