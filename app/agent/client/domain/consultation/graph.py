from __future__ import annotations

from typing import Any

from langgraph.constants import END, START
from langgraph.graph import StateGraph

from app.agent.client.domain.consultation.helpers import (
    DEFAULT_COMFORT_TEXT,
    DEFAULT_QUESTION_TEXT,
    build_text_result,
    collect_consultation_traces,
)
from app.agent.client.domain.consultation.nodes import (
    consultation_comfort_node,
    consultation_final_diagnosis_node,
    consultation_question_card_node,
    consultation_status_node,
    consultation_stream_response_node,
)
from app.agent.client.domain.consultation.state import (
    CONSULTATION_STATUS_COLLECTING,
    ConsultationState,
)


def _route_from_status(state: ConsultationState) -> str:
    """根据状态判断节点结果选择下一个 consultation 节点。"""

    if bool(state.get("should_enter_diagnosis", False)):
        return "final_diagnosis_node"
    return "collecting_fanout_node"


def _route_after_parallel_merge(state: ConsultationState) -> str:
    """根据并行问询结果决定是否继续进入最终诊断。"""

    if bool(state.get("should_enter_diagnosis", False)):
        return "final_diagnosis_node"
    return "collecting_response_node"


def consultation_collecting_fanout_node(state: ConsultationState) -> dict[str, Any]:
    """作为并行节点的 fanout 占位节点。"""

    _ = state
    return {}


def consultation_parallel_merge_node(state: ConsultationState) -> dict[str, Any]:
    """作为并行节点汇合后的条件判断占位节点。"""

    _ = state
    return {}


def consultation_collecting_response_node(state: ConsultationState) -> dict[str, Any]:
    """拼接 collecting 分支的最终文本。"""

    final_text = build_text_result(
        str(state.get("comfort_text") or ""),
        str(state.get("question_text") or ""),
    )
    if not final_text:
        final_text = build_text_result(DEFAULT_COMFORT_TEXT, DEFAULT_QUESTION_TEXT)
    return {
        "final_text": final_text,
        "consultation_status": CONSULTATION_STATUS_COLLECTING,
        "node_traces": collect_consultation_traces(state),
    }


def build_consultation_graph() -> Any:
    """构建 consultation 子图。"""

    graph = StateGraph(ConsultationState)

    graph.add_node("consultation_status_node", consultation_status_node)
    graph.add_node("collecting_fanout_node", consultation_collecting_fanout_node)
    graph.add_node("consultation_comfort_node", consultation_comfort_node)
    graph.add_node("consultation_question_card_node", consultation_question_card_node)
    graph.add_node("consultation_parallel_merge_node", consultation_parallel_merge_node)
    graph.add_node("collecting_response_node", consultation_collecting_response_node)
    graph.add_node("final_diagnosis_node", consultation_final_diagnosis_node)
    graph.add_node("consultation_stream_response_node", consultation_stream_response_node)

    graph.add_edge(START, "consultation_status_node")
    graph.add_conditional_edges(
        "consultation_status_node",
        _route_from_status,
        {
            "final_diagnosis_node": "final_diagnosis_node",
            "collecting_fanout_node": "collecting_fanout_node",
        },
    )
    graph.add_edge("collecting_fanout_node", "consultation_comfort_node")
    graph.add_edge("collecting_fanout_node", "consultation_question_card_node")
    graph.add_edge("consultation_comfort_node", "consultation_parallel_merge_node")
    graph.add_edge("consultation_question_card_node", "consultation_parallel_merge_node")
    graph.add_conditional_edges(
        "consultation_parallel_merge_node",
        _route_after_parallel_merge,
        {
            "final_diagnosis_node": "final_diagnosis_node",
            "collecting_response_node": "collecting_response_node",
        },
    )
    graph.add_edge("collecting_response_node", "consultation_stream_response_node")
    graph.add_edge("final_diagnosis_node", "consultation_stream_response_node")
    graph.add_edge("consultation_stream_response_node", END)

    return graph.compile()


# consultation 子图编译结果，供父图包装节点复用。
_CONSULTATION_GRAPH = build_consultation_graph()

__all__ = [
    "_CONSULTATION_GRAPH",
    "_route_after_parallel_merge",
    "_route_from_status",
    "build_consultation_graph",
    "consultation_collecting_fanout_node",
    "consultation_collecting_response_node",
    "consultation_parallel_merge_node",
    "consultation_stream_response_node",
]
