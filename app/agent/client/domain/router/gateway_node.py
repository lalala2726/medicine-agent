from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field, ValidationError

from app.agent.client.state import AgentState, ExecutionTraceState
from app.core.agent.agent_runtime import agent_invoke
from app.core.agent.agent_tool_trace import record_agent_trace
from app.core.agent.base_prompt_middleware import BasePromptMiddleware
from app.core.config_sync import AgentChatModelSlot, create_agent_chat_llm
from app.core.langsmith import traceable
from app.services.token_usage_service import append_trace_and_refresh_token_usage
from app.utils.prompt_utils import load_prompt

_GATEWAY_PROMPT = load_prompt("client/gateway_prompt.md")
_GATEWAY_ROUTE_FALLBACK: dict[str, Any] = {
    "route_targets": ["chat_agent"],
    "task_difficulty": "normal",
}
_ALLOWED_GATEWAY_TARGETS: tuple[str, ...] = (
    "chat_agent",
    "order_agent",
    "product_agent",
    "after_sale_agent",
)


class GatewayRoutingSchema(BaseModel):
    """Client gateway 路由结构化输出。"""

    route_targets: list[
        Literal[
            "chat_agent",
            "order_agent",
            "product_agent",
            "after_sale_agent",
        ]
    ] = Field(
        min_length=1,
        description=(
            "路由目标数组，仅允许 "
            "chat_agent/order_agent/product_agent/after_sale_agent"
        ),
    )
    task_difficulty: Literal["normal", "high"] = Field(
        description="任务难度，仅允许 normal 或 high",
    )


def _normalize_route_targets(route_targets: list[str]) -> list[str]:
    """规范化 client gateway 目标数组。"""

    normalized: list[str] = []
    for raw_target in route_targets:
        target = str(raw_target or "").strip()
        if not target:
            return []
        if target not in _ALLOWED_GATEWAY_TARGETS:
            return []
        if target in normalized:
            continue
        normalized.append(target)

    if len(normalized) != 1:
        return []
    return normalized


def _resolve_gateway_routing_result(raw_payload: Any) -> dict[str, Any]:
    """解析 client gateway 的结构化路由结果。"""

    if not isinstance(raw_payload, Mapping):
        return dict(_GATEWAY_ROUTE_FALLBACK)

    raw_messages = raw_payload.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        return dict(_GATEWAY_ROUTE_FALLBACK)

    last_message = raw_messages[-1]
    raw_content = getattr(last_message, "content", None)
    if not isinstance(raw_content, str):
        return dict(_GATEWAY_ROUTE_FALLBACK)

    try:
        parsed_json = json.loads(raw_content.strip())
    except json.JSONDecodeError:
        return dict(_GATEWAY_ROUTE_FALLBACK)
    if not isinstance(parsed_json, Mapping):
        return dict(_GATEWAY_ROUTE_FALLBACK)

    try:
        parsed = GatewayRoutingSchema.model_validate(parsed_json)
    except ValidationError:
        return dict(_GATEWAY_ROUTE_FALLBACK)

    normalized_targets = _normalize_route_targets(list(parsed.route_targets))
    if not normalized_targets:
        return dict(_GATEWAY_ROUTE_FALLBACK)
    return {
        "route_targets": normalized_targets,
        "task_difficulty": parsed.task_difficulty,
    }


@traceable(name="Client Assistant Gateway Router Node", run_type="chain")
def gateway_router(state: AgentState) -> dict[str, Any]:
    """执行 client gateway 路由节点。"""

    llm = create_agent_chat_llm(
        slot=AgentChatModelSlot.CLIENT_ROUTE,
        temperature=0.0,
        think=False,
    )
    llm_model_name = str(getattr(llm, "model_name", "") or "").strip()
    agent = create_agent(
        model=llm,
        system_prompt=SystemMessage(content=_GATEWAY_PROMPT),
        middleware=[BasePromptMiddleware()],
    )
    history_messages = list(state.get("history_messages") or [])
    result = agent_invoke(agent, history_messages[-20:])
    routing = _resolve_gateway_routing_result(result.payload)
    trace = record_agent_trace(
        payload=result.payload,
        input_messages=history_messages,
        fallback_text=result.content,
    )
    current_execution_traces = list(state.get("execution_traces") or [])
    trace_item = ExecutionTraceState(
        sequence=len(current_execution_traces) + 1,
        node_name="gateway_router",
        model_name=llm_model_name or str(trace.get("model_name") or "unknown"),
        status="success",
        output_text=str(trace.get("text") or ""),
        llm_usage_complete=bool(trace.get("is_usage_complete", False)),
        llm_token_usage=trace.get("usage"),
        tool_calls=[],
        node_context={
            "route_targets": list(routing.get("route_targets") or []),
            "task_difficulty": routing.get("task_difficulty"),
        },
    )
    execution_traces, token_usage = append_trace_and_refresh_token_usage(
        current_execution_traces,
        trace_item,
    )
    return {
        "routing": routing,
        "execution_traces": execution_traces,
        "token_usage": token_usage,
    }
