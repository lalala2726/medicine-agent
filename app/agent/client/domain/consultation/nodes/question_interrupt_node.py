from __future__ import annotations

from langgraph.types import interrupt

from app.agent.client.domain.consultation.helpers import (
    DEFAULT_QUESTION_OPTIONS,
    DEFAULT_QUESTION_REPLY_TEXT,
    DEFAULT_QUESTION_TEXT,
    append_followup_progress,
    append_resume_messages,
    append_trace_to_state,
    build_interrupt_payload,
    build_text_result,
    build_trace_item,
    resolve_consultation_outputs,
    resolve_consultation_progress,
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
        通过 `interrupt()` 挂起 consultation 追问，并在恢复后把问答和追问进度写回状态。

    参数说明：
        state (ConsultationState): 当前 consultation 状态。

    返回值：
        dict[str, object]: 恢复后需要写回的状态更新。

    异常说明：
        无；中断行为由 LangGraph runtime 接管。
    """

    outputs = resolve_consultation_outputs(state)
    progress = resolve_consultation_progress(state)
    response_text = str((outputs.get("response") or {}).get("text") or "").strip()
    question_section = outputs.get("question") or {}
    question_reply_text = str(question_section.get("reply_text") or "").strip() or DEFAULT_QUESTION_REPLY_TEXT
    question_text = str(question_section.get("question_text") or "").strip() or DEFAULT_QUESTION_TEXT
    raw_options = question_section.get("options")
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
    ai_reply_text = str(question_section.get("ai_reply_text") or "").strip() or build_text_result(
        response_text,
        question_reply_text,
    )
    history_messages = append_resume_messages(
        state=state,
        ai_reply_text=ai_reply_text,
        resume_text=resume_text,
    )
    consultation_progress = append_followup_progress(
        state=state,
        slot_key=str(progress.get("pending_slot_key") or ""),
        question_text=question_text,
        options=options,
        answer_text=resume_text,
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
            "slot_key": progress.get("pending_slot_key"),
        },
    )
    execution_traces, token_usage = append_trace_to_state(
        state=state,
        trace_item=interrupt_trace,
    )
    return {
        "consultation_status": CONSULTATION_STATUS_COLLECTING,
        "diagnosis_ready": False,
        "consultation_outputs": {
            "response": {"text": ""},
            "question": {
                "reply_text": "",
                "question_text": "",
                "options": [],
                "ai_reply_text": "",
            },
            "interrupt": {"payload": interrupt_payload},
        },
        "consultation_progress": consultation_progress,
        "history_messages": history_messages,
        "last_resume_text": resume_text,
        "interrupt_trace": interrupt_trace,
        "execution_traces": execution_traces,
        "token_usage": token_usage,
        "result": "",
        "messages": [],
    }


__all__ = [
    "consultation_question_interrupt_node",
]
