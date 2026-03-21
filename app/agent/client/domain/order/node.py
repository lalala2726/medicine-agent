from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.messages import AIMessage, SystemMessage

from app.agent.client.domain.order.tools import (
    check_order_cancelable,
    get_order_detail,
    get_order_shipping,
    get_order_timeline,
)
from app.agent.client.domain.tools.action_tools import open_user_order_list
from app.agent.client.state import AgentState, ExecutionTraceState
from app.core.agent.agent_event_bus import emit_answer_delta, emit_thinking_delta
from app.core.agent.agent_runtime import agent_stream
from app.core.agent.agent_tool_trace import record_agent_trace
from app.core.agent.base_prompt_middleware import BasePromptMiddleware
from app.core.config_sync import AgentChatModelSlot, create_agent_chat_llm
from app.core.langsmith import traceable
from app.services.token_usage_service import append_trace_and_refresh_token_usage
from app.utils.prompt_utils import append_current_time_to_prompt, load_prompt

_ORDER_SYSTEM_PROMPT = load_prompt("client/order_node_system_prompt.md")


@traceable(name="Client Assistant Order Agent Node", run_type="chain")
def order_agent(state: AgentState) -> dict[str, Any]:
    """执行 client 订单节点。"""

    history_messages = list(state.get("history_messages") or [])
    llm = create_agent_chat_llm(
        slot=AgentChatModelSlot.CLIENT_ORDER,
        temperature=1.0,
        think=False,
    )
    llm_model_name = str(getattr(llm, "model_name", "") or "").strip()
    agent = create_agent(
        model=llm,
        tools=[
            open_user_order_list,
            get_order_detail,
            get_order_shipping,
            get_order_timeline,
            check_order_cancelable,
        ],
        system_prompt=SystemMessage(
            content=append_current_time_to_prompt(_ORDER_SYSTEM_PROMPT)
        ),
        middleware=[
            BasePromptMiddleware(),
            ToolCallLimitMiddleware(thread_limit=5, run_limit=5),
        ],
    )
    stream_result = agent_stream(
        agent,
        history_messages,
        on_model_delta=emit_answer_delta,
        on_thinking_delta=emit_thinking_delta,
    )
    trace = record_agent_trace(
        payload=stream_result,
        input_messages=history_messages,
        fallback_text=str(stream_result.get("streamed_text") or ""),
    )
    current_execution_traces = list(state.get("execution_traces") or [])
    text = str(trace.get("text") or "").strip()
    trace_model_name = str(trace.get("model_name") or "").strip()
    trace_item = ExecutionTraceState(
        sequence=len(current_execution_traces) + 1,
        node_name="order_agent",
        model_name=llm_model_name or trace_model_name or "unknown",
        status="success",
        output_text=text,
        llm_usage_complete=bool(trace.get("is_usage_complete", False)),
        llm_token_usage=trace.get("usage"),
        tool_calls=list(trace.get("tool_calls") or []),
        node_context=None,
    )
    execution_traces, token_usage = append_trace_and_refresh_token_usage(
        current_execution_traces,
        trace_item,
    )
    return {
        "result": text,
        "messages": [AIMessage(content=text)],
        "execution_traces": execution_traces,
        "token_usage": token_usage,
    }
