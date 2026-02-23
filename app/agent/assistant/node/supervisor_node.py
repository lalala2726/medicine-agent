from __future__ import annotations

from typing import Any

from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.messages import AIMessage, SystemMessage

from app.agent.assistant.model_switch import model_switch
from app.utils.prompt_utils import load_prompt
from app.agent.assistant.state import AgentState, ExecutionTraceState
from app.agent.assistant.tools.analytics_tool import analytics_tool_agent
from app.agent.assistant.tools.base_tools import get_current_time
from app.agent.assistant.tools.chart_tool import chart_tool_agent
from app.agent.assistant.tools.order_tool import order_tool_agent
from app.agent.assistant.tools.product_tool import product_tool_agent
from app.core.agent.agent_event_bus import emit_answer_delta
from app.core.agent.agent_runtime import agent_stream
from app.core.agent.agent_tool_events import build_tool_status_middleware
from app.core.agent.agent_tool_trace import record_agent_trace
from app.core.langsmith import traceable
from app.core.llm import create_agent_instance
from app.services.token_usage_service import append_trace_and_refresh_token_usage

_BASE_PROMPT = load_prompt("assistant_base_prompt")
_SUPERVISOR_PROMPT = load_prompt("assistant_supervisor_system_prompt") + _BASE_PROMPT


@traceable(name="Supervisor Agent Node", run_type="chain")
def supervisor_agent(state: AgentState) -> dict[str, Any]:
    model_name = model_switch(state)
    history_messages = list(state.get("history_messages") or [])

    agent = create_agent_instance(
        model=model_name,
        llm_kwargs={"temperature": 1.3},
        system_prompt=SystemMessage(content=_SUPERVISOR_PROMPT),
        tools=[
            get_current_time,
            order_tool_agent,
            product_tool_agent,
            analytics_tool_agent,
            chart_tool_agent,
        ],
        middleware=[
            build_tool_status_middleware(),
            ToolCallLimitMiddleware(thread_limit=20, run_limit=10),
        ],
    )
    stream_result = agent_stream(
        agent,
        history_messages,
        on_model_delta=emit_answer_delta,
    )
    trace = record_agent_trace(
        payload=stream_result,
        input_messages=history_messages,
        fallback_text=str(stream_result.get("streamed_text") or ""),
    )
    text = str(trace.get("text") or "").strip()
    trace_item = ExecutionTraceState(
        node_name="supervisor_agent",
        model_name=model_name,
        output_text=text,
        llm_used=True,
        llm_usage_complete=bool(trace.get("is_usage_complete", False)),
        llm_token_usage=trace.get("usage"),
        tool_calls=list(trace.get("tool_calls") or []),
    )
    execution_traces, token_usage = append_trace_and_refresh_token_usage(
        state.get("execution_traces"),
        trace_item,
    )
    return {
        "result": text,
        "messages": [AIMessage(content=text)],
        "execution_traces": execution_traces,
        "token_usage": token_usage,
    }
