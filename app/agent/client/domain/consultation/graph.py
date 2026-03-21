from __future__ import annotations

from typing import Any

from langgraph.constants import END, START
from langgraph.graph import StateGraph

from app.agent.client.domain.consultation.helpers import (
    DEFAULT_COMFORT_TEXT,
    build_text_result,
    merge_parallel_round_traces,
)
from app.agent.client.domain.consultation.nodes import (
    consultation_comfort_node,
    consultation_final_diagnosis_node,
    consultation_question_interrupt_node,
    consultation_question_node,
)
from app.agent.client.domain.consultation.state import ConsultationState
from app.core.agent.langgraph_redis_checkpoint import _REDIS_CHECKPOINT_SAVER


def _route_from_entry(state: ConsultationState) -> str:
    """
    功能描述：
        根据 consultation 当前状态决定入口直接进入诊断还是 collecting 分支。

    参数说明：
        state (ConsultationState): 当前 consultation 状态。

    返回值：
        str: 下一跳节点名称。

    异常说明：
        无。
    """

    if bool(state.get("diagnosis_ready")):
        return "consultation_final_diagnosis_node"
    return "consultation_collecting_fanout_node"


def _route_after_parallel_merge(state: ConsultationState) -> str:
    """
    功能描述：
        根据追问节点结果决定 collecting 阶段后进入最终诊断还是 interrupt。

    参数说明：
        state (ConsultationState): 当前 consultation 状态。

    返回值：
        str: 下一跳节点名称。

    异常说明：
        无。
    """

    if bool(state.get("diagnosis_ready")):
        return "consultation_final_diagnosis_node"
    return "consultation_question_interrupt_node"


def consultation_collecting_fanout_node(_state: ConsultationState) -> dict[str, Any]:
    """
    功能描述：
        作为 collecting 阶段的扇出节点，触发 comfort 与 question 两个并行分支。

    参数说明：
        _state (ConsultationState): 当前 consultation 状态。

    返回值：
        dict[str, Any]: 空状态更新。

    异常说明：
        无。
    """

    return {}


def consultation_parallel_merge_node(state: ConsultationState) -> dict[str, Any]:
    """
    功能描述：
        合并 comfort/question 两条并行分支的文本与 trace。

    参数说明：
        state (ConsultationState): 当前 consultation 状态。

    返回值：
        dict[str, Any]: 合并后的状态更新。

    异常说明：
        无。
    """

    comfort_text = str(state.get("comfort_text") or "").strip() or DEFAULT_COMFORT_TEXT
    question_reply_text = str(state.get("question_reply_text") or "").strip()
    diagnosis_ready = bool(state.get("diagnosis_ready"))
    pending_ai_reply_text = (
        ""
        if diagnosis_ready
        else build_text_result(comfort_text, question_reply_text)
    )
    execution_traces, token_usage = merge_parallel_round_traces(state=state)
    return {
        "pending_ai_reply_text": pending_ai_reply_text,
        "execution_traces": execution_traces,
        "token_usage": token_usage,
        "result": "",
        "messages": [],
    }


def build_consultation_graph() -> Any:
    """
    功能描述：
        构建 consultation 子图。

    参数说明：
        无。

    返回值：
        Any: 编译后的 consultation graph。

    异常说明：
        无。
    """

    graph = StateGraph(ConsultationState)
    graph.add_node("consultation_collecting_fanout_node", consultation_collecting_fanout_node)
    graph.add_node("consultation_comfort_node", consultation_comfort_node)
    graph.add_node("consultation_question_node", consultation_question_node)
    graph.add_node("consultation_parallel_merge_node", consultation_parallel_merge_node)
    graph.add_node("consultation_question_interrupt_node", consultation_question_interrupt_node)
    graph.add_node("consultation_final_diagnosis_node", consultation_final_diagnosis_node)

    graph.add_conditional_edges(
        START,
        _route_from_entry,
        {
            "consultation_collecting_fanout_node": "consultation_collecting_fanout_node",
            "consultation_final_diagnosis_node": "consultation_final_diagnosis_node",
        },
    )
    graph.add_edge("consultation_collecting_fanout_node", "consultation_comfort_node")
    graph.add_edge("consultation_collecting_fanout_node", "consultation_question_node")
    graph.add_edge("consultation_comfort_node", "consultation_parallel_merge_node")
    graph.add_edge("consultation_question_node", "consultation_parallel_merge_node")
    graph.add_conditional_edges(
        "consultation_parallel_merge_node",
        _route_after_parallel_merge,
        {
            "consultation_question_interrupt_node": "consultation_question_interrupt_node",
            "consultation_final_diagnosis_node": "consultation_final_diagnosis_node",
        },
    )
    graph.add_edge("consultation_question_interrupt_node", "consultation_collecting_fanout_node")
    graph.add_edge("consultation_final_diagnosis_node", END)
    return graph.compile(checkpointer=_REDIS_CHECKPOINT_SAVER)


# consultation 子图编译结果，供父图包装节点与 resume 流程复用。
_CONSULTATION_GRAPH = build_consultation_graph()

__all__ = [
    "_CONSULTATION_GRAPH",
    "_route_after_parallel_merge",
    "_route_from_entry",
    "build_consultation_graph",
    "consultation_collecting_fanout_node",
    "consultation_parallel_merge_node",
]
