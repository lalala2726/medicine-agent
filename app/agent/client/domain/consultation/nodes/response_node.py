from __future__ import annotations

from langchain.agents.middleware import ToolCallLimitMiddleware

from app.agent.graph_tool import (
    query_disease_candidates_by_symptoms,
    query_disease_detail,
    search_symptom_candidates,
)
from app.agent.client.domain.consultation.helpers import (
    CONSULTATION_RESPONSE_MODEL_SLOT,
    DEFAULT_RESPONSE_TEXT,
    append_trace_to_state,
    build_consultation_input_messages,
    build_llm_agent,
    build_trace_item,
    resolve_consultation_route,
    resolve_natural_language_text,
)
from app.agent.client.domain.consultation.state import (
    CONSULTATION_STATUS_COMPLETED,
    ConsultationState,
)
from app.core.agent.agent_event_bus import emit_answer_delta, emit_thinking_delta
from app.core.agent.agent_runtime import agent_stream
from app.core.agent.agent_tool_trace import record_agent_trace
from app.core.langsmith import traceable
from app.utils.prompt_utils import load_prompt

# consultation 简单医学回应节点提示词。
SIMPLE_MEDICAL_RESPONSE_PROMPT = load_prompt(
    "client/consultation/simple_medical_response_system_prompt.md"
)
# consultation 诊断型医学回应节点提示词。
DIAGNOSTIC_RESPONSE_PROMPT = load_prompt(
    "client/consultation/diagnostic_response_system_prompt.md"
)
# consultation 医学回应节点可用的症状分析图谱工具。
CONSULTATION_RESPONSE_TOOLS = [
    search_symptom_candidates,
    query_disease_candidates_by_symptoms,
    query_disease_detail,
]
# consultation 医学回应节点单轮工具调用上限，避免单次回应做过多图谱检索。
CONSULTATION_RESPONSE_TOOL_CALL_LIMIT = ToolCallLimitMiddleware(thread_limit=4, run_limit=4)


@traceable(name="Client Consultation Response Node", run_type="chain")
def consultation_response_node(state: ConsultationState) -> dict[str, object]:
    """
    功能描述：
        根据 consultation 内路由结果输出医学回应文本；简单医学分支回答后直接结束，诊断型分支仅作为并行回应部分。

    参数说明：
        state (ConsultationState): 当前 consultation 状态。

    返回值：
        dict[str, object]: 当前节点写回的医学回应文本与 trace 信息。

    异常说明：
        无；节点执行异常由上层 workflow 统一处理。
    """

    consultation_route = resolve_consultation_route(state)
    next_action = str(consultation_route.get("next_action") or "").strip()
    prompt_text = (
        SIMPLE_MEDICAL_RESPONSE_PROMPT
        if next_action == "reply_only"
        else DIAGNOSTIC_RESPONSE_PROMPT
    )
    input_messages = build_consultation_input_messages(
        state=state,
        include_progress_context=True,
    )
    agent, llm_model_name = build_llm_agent(
        state=state,
        slot=CONSULTATION_RESPONSE_MODEL_SLOT,
        temperature=1.0,
        prompt_text=prompt_text,
        tools=CONSULTATION_RESPONSE_TOOLS,
        extra_middleware=[CONSULTATION_RESPONSE_TOOL_CALL_LIMIT],
    )
    stream_result = agent_stream(
        agent,
        input_messages,
        on_model_delta=emit_answer_delta,
        on_thinking_delta=emit_thinking_delta,
    )
    trace_payload = record_agent_trace(
        payload=stream_result,
        input_messages=input_messages,
        fallback_text=str(stream_result.get("streamed_text") or ""),
    )
    response_text = resolve_natural_language_text(
        trace_text=str(trace_payload.get("text") or ""),
        fallback_text=str(stream_result.get("streamed_text") or ""),
        default_text=DEFAULT_RESPONSE_TEXT,
    )
    response_trace = build_trace_item(
        node_name="consultation_response_node",
        llm_model_name=llm_model_name or str(trace_payload.get("model_name") or "unknown"),
        output_text=response_text,
        llm_usage_complete=bool(trace_payload.get("is_usage_complete", False)),
        llm_token_usage=trace_payload.get("usage"),
        tool_calls=list(trace_payload.get("tool_calls") or []),
        node_context={
            "next_action": next_action,
            "consultation_mode": consultation_route.get("consultation_mode"),
        },
    )

    if next_action == "reply_only":
        execution_traces, token_usage = append_trace_to_state(
            state=state,
            trace_item=response_trace,
        )
        return {
            "consultation_status": CONSULTATION_STATUS_COMPLETED,
            "diagnosis_ready": False,
            "consultation_outputs": {
                "response": {"text": response_text},
            },
            "response_trace": response_trace,
            "execution_traces": execution_traces,
            "token_usage": token_usage,
            "result": response_text,
            "messages": [],
        }

    return {
        "consultation_outputs": {
            "response": {"text": response_text},
        },
        "response_trace": response_trace,
        "result": "",
        "messages": [],
    }


__all__ = [
    "CONSULTATION_RESPONSE_TOOLS",
    "DIAGNOSTIC_RESPONSE_PROMPT",
    "SIMPLE_MEDICAL_RESPONSE_PROMPT",
    "consultation_response_node",
]
