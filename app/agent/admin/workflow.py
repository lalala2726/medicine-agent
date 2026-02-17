from __future__ import annotations

import json
from typing import Any

from langgraph.constants import END, START
from langgraph.graph import StateGraph

from app.agent.admin.history_utils import build_messages_with_history
from app.agent.admin.agent_state import AgentState
from app.agent.admin.dag_rules import EXECUTION_NODES, compute_planner_update
from app.agent.admin.node import (
    chat_agent,
    chart_agent,
    coordinator,
    excel_agent,
    order_agent,
    summary_agent,
)
from app.agent.admin.node.product_node import product_agent
from app.core.langsmith import traceable
from app.core.llm import create_chat_model

# 网关直达节点：仅保留订单、商品、聊天。
GATEWAY_DIRECT_NODES = (
    "order_agent",
    "product_agent",
)
GATEWAY_ROUTE_NODES = (*GATEWAY_DIRECT_NODES, "chat_agent", "coordinator_agent")
GATEWAY_ROUTE_MAP = {
    "order_agent": "order_agent",
    "product_agent": "product_agent",
    "chat_agent": "chat_agent",
    "coordinator_agent": "coordinator_agent",
    END: END,
}
PLANNER_ROUTE_MAP = {
    "order_agent": "order_agent",
    "excel_agent": "excel_agent",
    "chart_agent": "chart_agent",
    "summary_agent": "summary_agent",
    "product_agent": "product_agent",
    "chat_agent": "chat_agent",
    END: END,
}

GATEWAY_ROUTER_PROMPT = """
    你是药品商城后台的网关路由节点（gateway_router）。
    你的任务是根据用户请求，决定是直接交给单个业务节点，还是交给 coordinator_agent 协调多节点执行。

    可用节点与业务范围：
    1. order_agent
       - 订单域任务：订单查询、订单状态判断、订单信息核验、订单明细检索。
    2. product_agent
       - 商品域任务：商品信息查询、商品信息核验。
    3. chat_agent
        - 普通对话：非业务相关问题、咨询、闲聊。
    4. coordinator_agent
       - 协调域任务：多节点任务拆解、并行/串行编排、跨节点结果整合。
       - excel/chart/summary 相关任务必须路由到该节点，由 coordinator 决定计划。
       - 当请求涉及两个及以上业务域，或依赖关系复杂时，也必须路由到该节点。

    请严格输出 JSON 对象，且只输出 JSON，不要包含其他文字：
    {
      "route_target": "order_agent | product_agent | chat_agent | coordinator_agent",
      "difficulty": "simple | medium | complex"
    }

    路由规则：
    1. 如果单个业务节点即可完成，且不需要跨节点协作：
       - route_target 设置为对应业务节点
       - difficulty = simple
    2. 如果任务涉及多个业务域、需要拆解并行/串行步骤、或存在明显依赖关系：
       - route_target = coordinator_agent
       - difficulty 按复杂度设置为 medium 或 complex
    3. 如果意图不清晰，默认 route_target = chat_agent, difficulty = simple。
"""


def build_graph():
    """
    构建 admin assistant 的 LangGraph 工作流图。

    图结构说明：
    - `START -> gateway_router`
    - gateway 可直接路由到业务节点 / chat / coordinator
    - coordinator 生成计划后进入 planner
    - planner 按 DAG 规则选择下一批执行节点
    - 业务节点执行后回到 planner，直到无可执行节点并结束

    Returns:
        已编译的 LangGraph，可直接通过 `invoke/stream` 执行。
    """
    graph = StateGraph(AgentState)

    graph.add_node("gateway_router", gateway_router)
    graph.add_node("coordinator_agent", coordinator)
    graph.add_node("planner", planner)
    graph.add_node("chat_agent", chat_agent)
    graph.add_node("summary_agent", summary_agent)
    graph.add_node("order_agent", order_agent)
    graph.add_node("excel_agent", excel_agent)
    graph.add_node("chart_agent", chart_agent)
    graph.add_node("product_agent", product_agent)

    graph.add_edge(START, "gateway_router")
    graph.add_conditional_edges(
        "gateway_router",
        _route_from_gateway,
        GATEWAY_ROUTE_MAP,
    )

    graph.add_edge("coordinator_agent", "planner")
    graph.add_edge("chat_agent", END)

    for node in EXECUTION_NODES:
        graph.add_edge(node, "planner")

    graph.add_conditional_edges(
        "planner",
        _route_to_next_node,
        PLANNER_ROUTE_MAP,
    )
    return graph.compile()


def _normalize_difficulty(value: str) -> str:
    """
    规范化难度标识，限制在系统支持的枚举范围内。

    Args:
        value: 任意来源的难度值。

    Returns:
        `simple` / `medium` / `complex` 之一；非法值回退为 `medium`。
    """
    difficulty = (value or "simple").strip().lower()
    if difficulty in {"simple", "medium", "complex"}:
        return difficulty
    return "medium"


@traceable(name="Gateway Router Node", run_type="chain")
def gateway_router(state: AgentState) -> dict[str, Any]:
    """
    第一跳网关节点：判断直接执行还是进入 coordinator_agent。

    行为：
    - 调用轻量模型根据用户请求输出 `route_target + difficulty`
    - 失败时回退到 `coordinator_agent + medium`
    - 初始化 DAG 调度运行态字段，避免后续节点判空

    Args:
        state: 当前 Agent 状态，主要读取 `user_input` 与 `routing`。

    Returns:
        仅返回 `routing` 更新，包含：
        - `route_target`
        - `difficulty`
        - `stage_index/next_nodes/next_step_ids/current_step_ids/completed_step_ids/blocked_step_ids`
    """
    user_input = str(state.get("user_input") or "").strip()
    history_messages = list(state.get("history_messages") or [])
    routing = dict(state.get("routing") or {})

    route_target = "coordinator_agent"
    difficulty = "medium"
    if user_input or history_messages:
        try:
            model = create_chat_model(
                model="qwen-flash",
                max_tokens=1024,
                temperature=0,
                response_format={"type": "json_object"},
            )
            messages = build_messages_with_history(
                system_prompt=GATEWAY_ROUTER_PROMPT,
                history_messages=history_messages,
                fallback_user_input=user_input,
            )
            response = model.invoke(messages)
            raw = json.loads(str(response.content))
            route_target = str(raw["route_target"])
            difficulty = str(raw["difficulty"])
        except Exception:
            route_target = "coordinator_agent"
            difficulty = "medium"

    if route_target not in GATEWAY_ROUTE_NODES:
        route_target = "coordinator_agent"

    # 初始化 DAG 调度相关运行态字段，保证后续节点读状态时不判空。
    routing["route_target"] = route_target
    routing["difficulty"] = _normalize_difficulty(difficulty)
    routing.setdefault("stage_index", 0)
    routing.setdefault("next_nodes", [])
    routing.setdefault("next_step_ids", [])
    routing.setdefault("current_step_ids", [])
    routing.setdefault("completed_step_ids", [])
    routing.setdefault("blocked_step_ids", [])

    return {"routing": routing}


def _route_from_gateway(state: AgentState) -> str:
    """
    将 gateway_router 产出的 route_target 映射为图中的下一跳节点名。

    Args:
        state: 当前 Agent 状态，读取 `routing.route_target`。

    Returns:
        合法节点名；若无效则返回 `END`。
    """
    routing = state.get("routing") or {}
    route_target = routing.get("route_target")
    if route_target in GATEWAY_ROUTE_NODES:
        return route_target
    return END


@traceable(name="Planner Node", run_type="chain")
def planner(state: AgentState) -> dict[str, Any]:
    """
    Planner 仅作为 workflow 入口，具体调度规则统一下沉到 dag_rules。
    这样 workflow.py 只负责图结构编排，规则迭代集中在单一模块维护。

    Args:
        state: 当前 Agent 运行状态。

    Returns:
        由 `compute_planner_update` 返回的调度更新结果。
    """
    return compute_planner_update(state)


def _route_to_next_node(state: AgentState) -> str | list[str]:
    """
    根据 planner 结果决定图中的下一跳。

    Args:
        state: 当前 Agent 状态，读取 `routing.next_nodes`。

    Returns:
        - 无可执行节点时返回 `END`
        - 单节点时返回节点名字符串
        - 多节点并行时返回节点名列表
    """
    routing = state.get("routing") or {}
    raw_next_nodes = routing.get("next_nodes")
    if not isinstance(raw_next_nodes, list):
        raw_next_nodes = []

    allowed_nodes = set(EXECUTION_NODES) | {"chat_agent"}
    next_nodes = [node for node in raw_next_nodes if node in allowed_nodes]
    if not next_nodes:
        return END
    if len(next_nodes) == 1:
        return next_nodes[0]
    return next_nodes
