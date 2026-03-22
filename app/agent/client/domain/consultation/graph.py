from __future__ import annotations

from typing import Any

from langgraph.constants import END, START
from langgraph.graph import StateGraph

from app.agent.client.domain.consultation.helpers import (
    build_text_result,
    merge_parallel_round_traces,
    resolve_consultation_outputs,
    resolve_consultation_route,
)
from app.agent.client.domain.consultation.nodes import (
    consultation_final_diagnosis_node,
    consultation_question_interrupt_node,
    consultation_question_node,
    consultation_response_node,
    consultation_route_node,
)
from app.agent.client.domain.consultation.state import ConsultationState
from app.core.agent.langgraph_redis_checkpoint import _REDIS_CHECKPOINT_SAVER


def _route_after_consultation_route(state: ConsultationState) -> str:
    """
    功能描述：
        根据 consultation 子图内路由结果决定进入直接回答、继续追问还是最终诊断分支。

    参数说明：
        state (ConsultationState): 当前 consultation 状态。

    返回值：
        str: 下一跳节点名称。

    异常说明：
        无。
    """

    consultation_route = resolve_consultation_route(state)
    next_action = str(consultation_route.get("next_action") or "").strip()
    if next_action == "reply_only":
        return "response_node"
    if next_action == "final_diagnosis":
        return "final_diagnosis_node"
    return "collecting_fanout_node"


def _route_after_consultation_response(state: ConsultationState) -> str:
    """
    功能描述：
        根据 consultation 子图当前路由动作，判断医学回应节点之后是直接结束还是进入并行汇合。

    参数说明：
        state (ConsultationState): 当前 consultation 状态。

    返回值：
        str: 下一跳节点名称。

    异常说明：
        无。
    """

    consultation_route = resolve_consultation_route(state)
    if str(consultation_route.get("next_action") or "").strip() == "reply_only":
        return END
    return "parallel_merge_node"


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
        return "final_diagnosis_node"
    return "question_interrupt_node"


def consultation_collecting_fanout_node(_state: ConsultationState) -> dict[str, Any]:
    """
    功能描述：
        作为 collecting 阶段的扇出节点，触发 response 与 question 两个并行分支。

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
        合并 response/question 两条并行分支的文本与 trace，并组装追问阶段完整 AI 回复文本。

    参数说明：
        state (ConsultationState): 当前 consultation 状态。

    返回值：
        dict[str, Any]: 合并后的状态更新。

    异常说明：
        无。
    """

    consultation_outputs = resolve_consultation_outputs(state)
    response_text = str((consultation_outputs.get("response") or {}).get("text") or "").strip()
    question_reply_text = str((consultation_outputs.get("question") or {}).get("reply_text") or "").strip()
    diagnosis_ready = bool(state.get("diagnosis_ready"))
    ai_reply_text = (
        ""
        if diagnosis_ready
        else build_text_result(response_text, question_reply_text)
    )
    execution_traces, token_usage = merge_parallel_round_traces(state=state)
    return {
        "consultation_outputs": {
            "question": {"ai_reply_text": ai_reply_text},
        },
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
    graph.add_node("route_node", consultation_route_node)
    graph.add_node("collecting_fanout_node", consultation_collecting_fanout_node)
    graph.add_node("response_node", consultation_response_node)
    graph.add_node("question_node", consultation_question_node)
    graph.add_node("parallel_merge_node", consultation_parallel_merge_node)
    graph.add_node("question_interrupt_node", consultation_question_interrupt_node)
    graph.add_node("final_diagnosis_node", consultation_final_diagnosis_node)

    graph.add_edge(START, "route_node")
    graph.add_conditional_edges(
        "route_node",
        _route_after_consultation_route,
        {
            "response_node": "response_node",
            "collecting_fanout_node": "collecting_fanout_node",
            "final_diagnosis_node": "final_diagnosis_node",
        },
    )
    graph.add_conditional_edges(
        "response_node",
        _route_after_consultation_response,
        {
            END: END,
            "parallel_merge_node": "parallel_merge_node",
        },
    )
    graph.add_edge("collecting_fanout_node", "response_node")
    graph.add_edge("collecting_fanout_node", "question_node")
    graph.add_edge("question_node", "parallel_merge_node")
    graph.add_conditional_edges(
        "parallel_merge_node",
        _route_after_parallel_merge,
        {
            "question_interrupt_node": "question_interrupt_node",
            "final_diagnosis_node": "final_diagnosis_node",
        },
    )
    graph.add_edge("question_interrupt_node", "route_node")
    graph.add_edge("final_diagnosis_node", END)
    return graph.compile(checkpointer=_REDIS_CHECKPOINT_SAVER)


# consultation 子图编译结果，供父图包装节点与 resume 流程复用。
_CONSULTATION_GRAPH = build_consultation_graph()

__all__ = [
    "_CONSULTATION_GRAPH",
    "_route_after_consultation_response",
    "_route_after_consultation_route",
    "_route_after_parallel_merge",
    "build_consultation_graph",
    "consultation_collecting_fanout_node",
    "consultation_parallel_merge_node",
]
