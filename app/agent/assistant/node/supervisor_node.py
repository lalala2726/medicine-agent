from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, SystemMessage

from app.agent.assistant.model_switch import model_switch
from app.utils.prompt_utils import load_prompt
from app.agent.assistant.state import AgentState, ExecutionTraceState
from app.agent.assistant.tools.analytics_tool import analytics_tool_agent
from app.agent.assistant.tools.chart_tool import chart_tool_agent
from app.agent.assistant.tools.order_tool import order_tool_agent
from app.agent.assistant.tools.product_tool import product_tool_agent
from app.core.agent_trace import run_agent_with_trace
from app.core.langsmith import traceable
from app.core.llm import create_agent_instance, create_chat_model
from app.services.token_usage_service import append_trace_and_refresh_token_usage

_BASE_PROMPT = load_prompt("assistant_base_prompt")
_SUPERVISOR_PROMPT = load_prompt("assistant_supervisor_system_prompt") + _BASE_PROMPT


@traceable(name="Supervisor Agent Node", run_type="chain")
def supervisor_agent(state: AgentState) -> dict[str, Any]:
    model_name = model_switch(state)

    llm = create_chat_model(
        model=model_name,
        temperature=1.3,
    )
    history_messages = list(state.get("history_messages") or [])

    agent = create_agent_instance(
        llm=llm,
        system_prompt=SystemMessage(content=_SUPERVISOR_PROMPT),
        tools=[order_tool_agent, product_tool_agent, analytics_tool_agent, chart_tool_agent],
    )
    trace = run_agent_with_trace(agent, history_messages)
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
