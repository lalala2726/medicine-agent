from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, SystemMessage

from app.agent.assistant.state import AgentState, ExecutionTraceState
from app.agent.assistant.tools import get_safe_user_info
from app.agent.assistant.tools.base_tools import get_current_time
from app.core.agent.agent_event_bus import emit_answer_delta, emit_thinking_delta
from app.core.agent.agent_runtime import agent_stream
from app.core.agent.agent_tool_trace import record_agent_trace
from app.core.agent.base_prompt_middleware import BasePromptMiddleware
from app.core.langsmith import traceable
from app.core.llms import LlmProvider, create_chat_model
from app.core.skill import SkillMiddleware
from app.services.token_usage_service import append_trace_and_refresh_token_usage
from app.utils.prompt_utils import load_prompt

_CHAT_SYSTEM_PROMPT = load_prompt("assistant/chat_system_prompt.md")


@traceable(name="Supervisor Chat Agent Node", run_type="chain")
def chat_agent(state: AgentState) -> dict[str, Any]:
    """
    执行 Chat 子代理节点，并透传思考流。

    Args:
        state: LangGraph 节点状态，包含历史消息与执行追踪信息。

    Returns:
        dict[str, Any]: 节点输出状态增量，包含 `result/messages/execution_traces/token_usage`。
    """

    model_name = "qwen3.5-plus"
    history_messages = list(state.get("history_messages") or [])
    # 占位开关：当前固定开启 Think，后续处理逻辑由调用方继续扩展。
    enable_think = True
    llm_kwargs: dict[str, Any] = {"temperature": 1.3}
    if enable_think:
        llm_kwargs["extra_body"] = {"enable_thinking": True}
    tools = [
        get_current_time,
        get_safe_user_info
    ]
    llm = create_chat_model(
        model="qwen3.5-plus",
        provider=LlmProvider.ALIYUN,
        **llm_kwargs,
    )
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
        on_thinking_delta=emit_thinking_delta if enable_think else None,
    )

    trace = record_agent_trace(
        payload=stream_result,
        input_messages=history_messages,
        fallback_text=str(stream_result.get("streamed_text") or ""),
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
