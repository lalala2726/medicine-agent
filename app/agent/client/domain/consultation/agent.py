from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.messages import AIMessage

from app.agent.client.domain.consultation.graph import _CONSULTATION_GRAPH
from app.agent.client.domain.consultation.helpers import collect_consultation_traces, resequence_traces
from app.agent.client.domain.consultation.state import (
    CONSULTATION_STATUS_COLLECTING,
    ConsultationState,
)
from app.agent.client.state import AgentState
from app.core.langsmith import traceable
from app.services.token_usage_service import build_token_usage_from_execution_traces


@traceable(name="Client Consultation Agent Node", run_type="chain")
def consultation_agent(state: AgentState) -> dict[str, Any]:
    """执行 client consultation 子图包装节点。"""

    history_messages = list(state.get("history_messages") or [])
    current_execution_traces = list(state.get("execution_traces") or [])
    routing = state.get("routing")
    task_difficulty = (
        str(routing.get("task_difficulty") or "").strip().lower()
        if isinstance(routing, Mapping)
        else ""
    )
    consultation_state = ConsultationState(
        history_messages=history_messages,
        task_difficulty=task_difficulty or "normal",
        consultation_status=CONSULTATION_STATUS_COLLECTING,
        should_enter_diagnosis=False,
        comfort_text="",
        question_text="",
        final_text="",
        recommended_product_ids=[],
    )
    graph_result = _CONSULTATION_GRAPH.invoke(consultation_state)
    if not isinstance(graph_result, Mapping):
        graph_result = consultation_state

    final_text = str(graph_result.get("final_text") or "").strip()
    subgraph_traces = collect_consultation_traces(ConsultationState(**dict(graph_result)))
    execution_traces = resequence_traces(
        existing_traces=current_execution_traces,
        appended_traces=subgraph_traces,
    )
    token_usage = build_token_usage_from_execution_traces(execution_traces)

    return {
        "result": final_text,
        "messages": [AIMessage(content=final_text)],
        "execution_traces": execution_traces,
        "token_usage": token_usage,
    }


__all__ = [
    "consultation_agent",
]
