from __future__ import annotations

from typing import Any, Mapping

from langchain_core.messages import AIMessage, SystemMessage

from app.agent.assistant.tools.base_tools import get_current_time
from app.agent.assistant.model_switch import model_switch
from app.utils.prompt_utils import load_prompt
from app.agent.assistant.state import AgentState, ExecutionTraceState
from app.core.agent.agent_runtime import agent_invoke
from app.core.agent.agent_tool_trace import record_agent_trace
from app.core.langsmith import traceable
from app.core.llm import create_agent_instance
from app.services.token_usage_service import append_trace_and_refresh_token_usage

_BASE_PROMPT = load_prompt("assistant_base_prompt")
_CHAT_SYSTEM_PROMPT = load_prompt("assistant_chat_system_prompt") + _BASE_PROMPT


@traceable(name="Supervisor Chat Agent Node", run_type="chain")
def chat_agent(state: AgentState) -> dict[str, Any]:
    model_name = model_switch(state)
    history_messages = list(state.get("history_messages") or [])
    tools = [
        get_current_time
    ]
    agent = create_agent_instance(
        model=model_name,
        tools=tools,
        llm_kwargs={"temperature": 1.3},
        system_prompt=SystemMessage(content=_CHAT_SYSTEM_PROMPT),
    )
    result = agent_invoke(agent, history_messages)
    fallback_text = ""
    if isinstance(result, Mapping):
        fallback_text = str(result.get("output") or result.get("text") or "")
    trace = record_agent_trace(
        payload=result,
        input_messages=history_messages,
        fallback_text=fallback_text,
    )
    text = str(trace.get("text") or "").strip()
    trace_item = ExecutionTraceState(
        node_name="chat_agent",
        model_name=model_name,
        output_text=text,
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
        "result": text,
        "messages": [AIMessage(content=text)],
        "execution_traces": execution_traces,
        "token_usage": token_usage,
    }
