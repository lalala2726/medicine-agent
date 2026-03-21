from __future__ import annotations

from langgraph.types import interrupt

from app.agent.client.domain.consultation.helpers import (
    DEFAULT_QUESTION_OPTIONS,
    DEFAULT_QUESTION_REPLY_TEXT,
    DEFAULT_QUESTION_TEXT,
    append_resume_messages,
    append_trace_to_state,
    build_interrupt_payload,
    build_text_result,
    build_trace_item,
    resolve_resume_text,
)
from app.agent.client.domain.consultation.state import (
    CONSULTATION_STATUS_COLLECTING,
    ConsultationState,
)
from app.core.langsmith import traceable


@traceable(name="Client Consultation Question Interrupt Node", run_type="chain")
def consultation_question_interrupt_node(state: ConsultationState) -> dict[str, object]:
    """
    功能描述：
        通过 `interrupt()` 挂起 consultation 追问，并在恢复后把问答写回历史。

    参数说明：
        state (ConsultationState): 当前 consultation 状态。

    返回值：
        dict[str, object]: 恢复后需要写回的状态更新。

    异常说明：
        无；中断行为由 LangGraph runtime 接管。
    """

    question_reply_text = str(state.get("question_reply_text") or "").strip() or DEFAULT_QUESTION_REPLY_TEXT
    question_text = str(state.get("pending_question_text") or "").strip() or DEFAULT_QUESTION_TEXT
    raw_options = state.get("pending_question_options")
    options = (
        [str(item).strip() for item in raw_options if str(item).strip()]
        if isinstance(raw_options, list)
        else []
    )
    if len(options) < 2:
        options = list(DEFAULT_QUESTION_OPTIONS)

    interrupt_payload = build_interrupt_payload(
        reply_text=question_reply_text,
        question_text=question_text,
        options=options,
    )
    resume_value = interrupt(interrupt_payload)
    resume_text = resolve_resume_text(resume_value)
    pending_ai_reply_text = str(state.get("pending_ai_reply_text") or "").strip() or build_text_result(
        str(state.get("comfort_text") or "").strip(),
        question_reply_text,
    )
    history_messages = append_resume_messages(
        state=state,
        ai_reply_text=pending_ai_reply_text,
        resume_text=resume_text,
    )

    interrupt_trace = build_trace_item(
        node_name="consultation_question_interrupt_node",
        llm_model_name="interrupt",
        output_text=question_reply_text,
        llm_usage_complete=True,
        llm_token_usage={
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        tool_calls=[],
        node_context={
            "reply_text": question_reply_text,
            "question_text": question_text,
            "options": options,
            "resume_text": resume_text,
        },
    )
    execution_traces, token_usage = append_trace_to_state(
        state=state,
        trace_item=interrupt_trace,
    )
    return {
        "consultation_status": CONSULTATION_STATUS_COLLECTING,
        "diagnosis_ready": False,
        "comfort_text": "",
        "question_reply_text": "",
        "pending_question_text": "",
        "pending_question_options": [],
        "pending_ai_reply_text": "",
        "history_messages": history_messages,
        "interrupt_payload": interrupt_payload,
        "last_resume_text": resume_text,
        "interrupt_trace": interrupt_trace,
        "execution_traces": execution_traces,
        "token_usage": token_usage,
    }


__all__ = [
    "consultation_question_interrupt_node",
]
