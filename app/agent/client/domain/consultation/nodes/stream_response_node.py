from __future__ import annotations

from typing import Any

from app.agent.client.domain.consultation.helpers import (
    emit_consultation_answer_deltas,
    resolve_stream_response_text,
)
from app.agent.client.domain.consultation.state import ConsultationState
from app.core.langsmith import traceable


@traceable(name="Client Consultation Stream Response Node", run_type="chain")
def consultation_stream_response_node(state: ConsultationState) -> dict[str, Any]:
    """按统一出口规则流式发送 consultation 最终文本。"""

    final_text = resolve_stream_response_text(state)
    emit_consultation_answer_deltas(final_text)
    return {
        "final_text": final_text,
    }


__all__ = [
    "consultation_stream_response_node",
]
