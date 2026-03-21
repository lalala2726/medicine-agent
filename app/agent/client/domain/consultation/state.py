from __future__ import annotations

from typing import Literal, TypedDict

from app.agent.client.state import ChatHistoryMessage, ExecutionTraceState

# consultation 子图进行中状态。
CONSULTATION_STATUS_COLLECTING: Literal["collecting"] = "collecting"
# consultation 子图已具备诊断条件状态。
CONSULTATION_STATUS_COMPLETED: Literal["completed"] = "completed"
# consultation 子图允许的阶段值。
ConsultationStatusValue = Literal["collecting", "completed"]


class ConsultationState(TypedDict, total=False):
    """Client 病情咨询子图状态。"""

    history_messages: list[ChatHistoryMessage]
    task_difficulty: str
    consultation_status: ConsultationStatusValue
    should_enter_diagnosis: bool
    comfort_text: str
    question_text: str
    final_text: str
    recommended_product_ids: list[int]
    node_traces: list[ExecutionTraceState]
    status_trace: ExecutionTraceState
    comfort_trace: ExecutionTraceState
    question_trace: ExecutionTraceState
    diagnosis_trace: ExecutionTraceState


__all__ = [
    "CONSULTATION_STATUS_COLLECTING",
    "CONSULTATION_STATUS_COMPLETED",
    "ConsultationState",
    "ConsultationStatusValue",
]
