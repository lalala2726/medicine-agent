from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import SystemMessage

from app.utils.prompt_utils import load_prompt
from app.agent.assistant.state import AgentState, ExecutionTraceState, GatewayRoutingState
from app.core.langsmith import traceable
from app.core.llm import create_chat_model
from app.services.token_usage_service import append_trace_and_refresh_token_usage
from app.utils.streaming_utils import (
    invoke_with_trace,
)

_GATEWAY_PROMPT = load_prompt("assistant_gateway_prompt")


@traceable(name="Supervisor Gateway Router Node", run_type="chain")
def gateway_router(state: AgentState) -> dict[str, Any]:
    llm = create_chat_model(
        model="qwen-flash",
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    history_messages = list(state.get("history_messages") or [])

    messages = [SystemMessage(content=_GATEWAY_PROMPT), *history_messages]

    trace = invoke_with_trace(llm, messages)
    raw_content = trace.get("raw_content")

    parsed: dict[str, Any] = {}
    if isinstance(raw_content, dict):
        parsed = raw_content
    else:
        raw_text = str(trace.get("text") or "").strip()
        try:
            candidate = json.loads(raw_text)
            if isinstance(candidate, dict):
                parsed = candidate
        except json.JSONDecodeError:
            parsed = {}

    route_target = str(parsed.get("route_target") or "").strip()
    if route_target not in {"chat_agent", "supervisor_agent"}:
        route_target = "chat_agent"

    task_difficulty = str(parsed.get("task_difficulty") or "").strip().lower()
    if task_difficulty not in {"simple", "normal", "complex"}:
        task_difficulty = "normal"

    routing: GatewayRoutingState = {
        "route_target": route_target,
        "task_difficulty": task_difficulty,
    }

    trace_item = ExecutionTraceState(
        node_name="gateway_router",
        model_name=str(trace.get("model_name") or "unknown"),
        output_text=str(trace.get("text") or ""),
        llm_used=True,
        llm_usage_complete=bool(trace.get("is_usage_complete", False)),
        llm_token_usage=trace.get("usage"),
        tool_calls=[],
    )
    execution_traces, token_usage = append_trace_and_refresh_token_usage(
        state.get("execution_traces"),
        trace_item,
    )
    return {
        "routing": routing,
        "execution_traces": execution_traces,
        "token_usage": token_usage,
    }
