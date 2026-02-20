from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.admin.agent_utils import build_execution_trace_update, serialize_messages
from app.agent.admin.state import AgentState
from app.core.langsmith import traceable
from app.core.llm import create_chat_model

# Gateway 可路由的目标节点白名单，确保路由输出可控。
_ALLOWED_ROUTE_TARGETS = {
    "order_agent",
    "product_agent",
    "chat_agent",
    "supervisor_agent",
}

_GATEWAY_PROMPT = """
你是药品商城后台助手的前置意图网关（Intent Gateway）。

输出要求：
1. 只输出 JSON；
2. 仅允许一个字段 route_target；
3. route_target 仅允许：order_agent、product_agent、chat_agent、supervisor_agent。

路由规则：
1. 单域且简单（纯订单或纯商品） -> order_agent 或 product_agent；
2. 闲聊、寒暄、无明确业务动作 -> chat_agent；
3. 跨域、多步骤、依赖上下文决策、复杂任务 -> supervisor_agent。
"""


def _resolve_mode(route_target: str) -> str:
    """
    根据 gateway 路由目标推导执行模式。

    Args:
        route_target: gateway 产出的目标节点名。

    Returns:
        str: 模式字符串：
            - `fast_lane`: 订单/商品直达；
            - `chat`: 闲聊节点；
            - `supervisor_loop`: 进入动态主管慢车道。
    """

    if route_target in {"order_agent", "product_agent"}:
        return "fast_lane"
    if route_target == "chat_agent":
        return "chat"
    return "supervisor_loop"


@traceable(name="Supervisor Gateway Router Node", run_type="chain")
def gateway_router(state: AgentState) -> dict[str, Any]:
    """
    执行前置意图网关路由。

    Args:
        state: 当前图状态，主要读取 `user_input` 与 `messages`（用于轻量历史辅助判断）。

    Returns:
        dict[str, Any]: gateway 的状态更新字典，字段说明如下：
            - `routing.route_target`: 路由目标节点；
            - `routing.mode`: 执行模式（fast_lane/chat/supervisor_loop）；
            - `routing.turn`: 初始化为 0；
            - `routing.finished`: 初始化为 `False`；
            - `next_node`: 与 route_target 保持一致；
            - `context`: 初始化共享上下文（含调用计数、agent_outputs、原始输入）；
            - `execution_traces`: 路由决策追踪。
    """

    user_input = str(state.get("user_input") or "").strip()
    history_messages = list(state.get("messages") or [])
    input_messages: list[Any] = [SystemMessage(content=_GATEWAY_PROMPT)]
    if history_messages:
        input_messages.extend(history_messages[-8:])
    elif user_input:
        input_messages.append(HumanMessage(content=user_input))

    route_target = "supervisor_agent"
    model_name = "qwen-flash"
    try:
        llm = create_chat_model(
            model=model_name,
            temperature=0,
            response_format={"type": "json_object"},
        )
        response = llm.invoke(input_messages)
        payload = json.loads(str(response.content))
        candidate = str(payload.get("route_target") or "").strip()
        if candidate in _ALLOWED_ROUTE_TARGETS:
            route_target = candidate
    except Exception:
        route_target = "supervisor_agent"

    mode = _resolve_mode(route_target)
    update: dict[str, Any] = {
        "routing": {
            "route_target": route_target,
            "mode": mode,
            "turn": 0,
            "finished": False,
        },
        "next_node": route_target,
        "context": {
            "node_call_counts": {},
            "agent_outputs": {},
            "original_user_input": user_input,
        },
    }
    update.update(
        build_execution_trace_update(
            node_name="gateway_router",
            model_name=model_name,
            input_messages=serialize_messages(input_messages),
            output_text=json.dumps(update, ensure_ascii=False, default=str),
            tool_calls=[],
        )
    )
    return update
