from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, SystemMessage

from app.agent.assistant.state import AgentState, ExecutionTraceState
from app.agent.assistant.domain.common.tools import get_current_time
from app.agent.assistant.domain.common.tools import get_safe_user_info
from app.core.agent.agent_event_bus import emit_answer_delta
from app.core.agent.agent_runtime import agent_stream
from app.core.agent.agent_tool_trace import record_agent_trace
from app.core.agent.base_prompt_middleware import BasePromptMiddleware
from app.core.langsmith import traceable
from app.core.llms import create_chat_model
from app.core.agent.skill import SkillMiddleware
from app.services.token_usage_service import append_trace_and_refresh_token_usage
from app.utils.prompt_utils import load_prompt

_CHAT_SYSTEM_PROMPT = load_prompt("assistant/chat_system_prompt.md")


@traceable(name="Assistant Chat Agent Node", run_type="chain")
def chat_agent(state: AgentState) -> dict[str, Any]:
    """
    执行 Chat 节点，并透传思考流。

    Args:
        state: LangGraph 节点状态，包含历史消息与执行追踪信息。

    Returns:
        dict[str, Any]: 节点输出状态增量，包含 `result/messages/execution_traces/token_usage`。
    """

    history_messages = list(state.get("history_messages") or [])
    tools = [
        get_current_time,
        get_safe_user_info
    ]
    llm = create_chat_model(
        temperature=1.3,
    )
    llm_model_name = str(getattr(llm, "model_name", "") or "").strip()
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SystemMessage(content=_CHAT_SYSTEM_PROMPT),
        middleware=[
            BasePromptMiddleware(),
            SkillMiddleware(scope="chat"),
        ]
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
    current_execution_traces = list(state.get("execution_traces") or [])
    text = str(trace.get("text") or "").strip()
    trace_model_name = str(trace.get("model_name") or "").strip()
    trace_item = ExecutionTraceState(
        sequence=len(current_execution_traces) + 1,
        node_name="chat_agent",
        model_name=llm_model_name or trace_model_name or "unknown",
        status="success",
        output_text=text,
        llm_usage_complete=bool(trace.get("is_usage_complete", False)),
        llm_token_usage=trace.get("usage"),
        tool_calls=[],
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
