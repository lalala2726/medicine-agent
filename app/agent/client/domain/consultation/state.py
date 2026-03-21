from __future__ import annotations

from typing import Any, Literal, TypedDict

from app.agent.client.state import (
    ChatHistoryMessage,
    ExecutionTraceState,
    TokenUsageState,
)

# consultation 子图进行中状态。
CONSULTATION_STATUS_COLLECTING: Literal["collecting"] = "collecting"
# consultation 子图已完成最终诊断状态。
CONSULTATION_STATUS_COMPLETED: Literal["completed"] = "completed"
# consultation 子图允许的阶段值。
ConsultationStatusValue = Literal["collecting", "completed"]


class ConsultationInterruptPayload(TypedDict, total=False):
    """consultation 追问中断负载。"""

    kind: str
    reply_text: str
    question_text: str
    options: list[str]


class ConsultationState(TypedDict, total=False):
    """Client 病情咨询子图状态。"""

    history_messages: list[ChatHistoryMessage]
    task_difficulty: str
    consultation_status: ConsultationStatusValue
    diagnosis_ready: bool
    comfort_text: str
    question_reply_text: str
    pending_question_text: str
    pending_question_options: list[str]
    pending_ai_reply_text: str
    final_text: str
    execution_traces: list[ExecutionTraceState]
    token_usage: TokenUsageState | None
    comfort_trace: ExecutionTraceState
    question_trace: ExecutionTraceState
    diagnosis_trace: ExecutionTraceState
    interrupt_trace: ExecutionTraceState
    interrupt_payload: ConsultationInterruptPayload | None
    last_resume_text: str
    result: str
    messages: list[Any]


__all__ = [
    "CONSULTATION_STATUS_COLLECTING",
    "CONSULTATION_STATUS_COMPLETED",
    "ConsultationInterruptPayload",
    "ConsultationState",
    "ConsultationStatusValue",
]
