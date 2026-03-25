from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.messages import AIMessage, SystemMessage

from app.agent.client.domain.diagnosis.tools import (
    query_disease_candidates_by_symptoms,
    query_disease_details,
    query_followup_symptom_candidates,
    search_symptom_candidates,
    send_consultation_questionnaire_card,
)
from app.agent.client.domain.diagnosis.tools.cache import (
    bind_current_diagnosis_tool_cache_conversation,
    render_diagnosis_tool_cache_prompt,
    reset_current_diagnosis_tool_cache_conversation,
)
from app.agent.client.state import AgentState, ExecutionTraceState
from app.core.agent.agent_event_bus import emit_answer_delta, emit_thinking_delta
from app.core.agent.agent_runtime import agent_stream
from app.core.agent.agent_tool_events import build_tool_status_middleware
from app.core.agent.agent_tool_trace import record_agent_trace
from app.core.agent.base_prompt_middleware import BasePromptMiddleware
from app.core.config_sync import AgentChatModelSlot, create_agent_chat_llm
from app.core.langsmith import traceable
from app.services.token_usage_service import append_trace_and_refresh_token_usage
from app.utils.prompt_utils import append_current_time_to_prompt, load_prompt

# 诊断节点固定系统提示词。
_DIAGNOSIS_SYSTEM_PROMPT_TEMPLATE = SystemMessagePromptTemplate.from_template(
    load_prompt("client/diagnosis_node_system_prompt.md"),
)


@traceable(name="Client Assistant Diagnosis Agent Node", run_type="chain")
def diagnosis_agent(state: AgentState) -> dict[str, Any]:
    """执行 client 诊断节点。

    Args:
        state: 当前 client agent 工作流状态。

    Returns:
        dict[str, Any]: 诊断节点输出结果、消息与执行轨迹。
    """

    conversation_uuid = str(state.get("conversation_uuid") or "").strip()
    history_messages = list(state.get("history_messages") or [])
    diagnosis_tool_cache_prompt = render_diagnosis_tool_cache_prompt(conversation_uuid)
    diagnosis_system_message = _DIAGNOSIS_SYSTEM_PROMPT_TEMPLATE.format(
        tool_cache=diagnosis_tool_cache_prompt,
    )
    llm = create_agent_chat_llm(
        slot=AgentChatModelSlot.CLIENT_CONSULTATION_FINAL_DIAGNOSIS,
        temperature=1.0,
        think=False,
    )
    llm_model_name = str(getattr(llm, "model_name", "") or "").strip()
    agent = create_agent(
        model=llm,
        tools=[
            search_symptom_candidates,
            query_disease_candidates_by_symptoms,
            query_followup_symptom_candidates,
            send_consultation_questionnaire_card,
            query_disease_details,
        ],
        system_prompt=SystemMessage(
            content=append_current_time_to_prompt(
                str(diagnosis_system_message.content or "")
            )
        ),
        middleware=[
            BasePromptMiddleware(),
            build_tool_status_middleware(),
            ToolCallLimitMiddleware(thread_limit=20, run_limit=20),
        ],
    )
    cache_token = bind_current_diagnosis_tool_cache_conversation(conversation_uuid)
    try:
        stream_result = agent_stream(
            agent,
            history_messages,
            on_model_delta=emit_answer_delta,
            on_thinking_delta=emit_thinking_delta,
        )
    finally:
        reset_current_diagnosis_tool_cache_conversation(cache_token)
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
        node_name="diagnosis_agent",
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
