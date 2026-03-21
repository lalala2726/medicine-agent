from collections.abc import Mapping
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from app.agent.client.domain.consultation.graph import _CONSULTATION_GRAPH
from app.agent.client.domain.consultation.helpers import (
    CONSULTATION_CHECKPOINT_NAMESPACE,
    build_consultation_graph_config,
    build_default_consultation_outputs,
    build_default_consultation_progress,
    build_default_consultation_route,
    resolve_consultation_result_text,
)
from app.agent.client.domain.consultation.state import (
    CONSULTATION_STATUS_COLLECTING,
    ConsultationState,
)
from app.agent.client.state import AgentState
from app.core.langsmith import traceable
from app.services.token_usage_service import build_token_usage_from_execution_traces


def _build_consultation_initial_state(state: AgentState) -> ConsultationState:
    """
    功能描述：
        将父图 AgentState 映射为 consultation 子图初始状态。

    参数说明：
        state (AgentState): 父图输入状态。

    返回值：
        ConsultationState: consultation 子图初始状态。

    异常说明：
        无。
    """

    history_messages = list(state.get("history_messages") or [])
    routing = state.get("routing")
    task_difficulty = (
        str(routing.get("task_difficulty") or "").strip().lower()
        if isinstance(routing, Mapping)
        else ""
    )
    return ConsultationState(
        history_messages=history_messages,
        task_difficulty=task_difficulty or "normal",
        consultation_status=CONSULTATION_STATUS_COLLECTING,
        diagnosis_ready=False,
        consultation_route=build_default_consultation_route(),
        consultation_progress=build_default_consultation_progress(),
        consultation_outputs=build_default_consultation_outputs(),
        execution_traces=[],
        token_usage=None,
        last_resume_text="",
        result="",
        messages=[],
    )


def has_pending_consultation_interrupt(*, conversation_uuid: str) -> bool:
    """
    功能描述：
        判断某个会话是否存在 consultation 可恢复中断。

    参数说明：
        conversation_uuid (str): 会话 UUID。

    返回值：
        bool: 存在可恢复中断时返回 `True`。

    异常说明：
        无；读取 checkpoint 失败时返回 `False`。
    """

    normalized_uuid = conversation_uuid.strip()
    if not normalized_uuid:
        return False

    try:
        graph_state = _CONSULTATION_GRAPH.get_state(
            {
                "configurable": {
                    "thread_id": normalized_uuid,
                    "checkpoint_ns": CONSULTATION_CHECKPOINT_NAMESPACE,
                }
            }
        )
    except Exception:
        return False

    interrupts = getattr(graph_state, "interrupts", ())
    return bool(interrupts)


@traceable(name="Client Consultation Agent Node", run_type="chain")
def consultation_agent(
        state: AgentState,
        config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """
    功能描述：
        执行 client consultation 子图包装节点。

    参数说明：
        state (AgentState): 父图输入状态。
        config (RunnableConfig | None): 父图 runnable config。

    返回值：
        dict[str, Any]: 映射回父图的状态更新。

    异常说明：
        无；子图异常由上层 workflow 统一处理。
    """

    consultation_state = _build_consultation_initial_state(state)
    graph_result = _CONSULTATION_GRAPH.invoke(
        consultation_state,
        config=build_consultation_graph_config(config),
    )
    if not isinstance(graph_result, Mapping):
        graph_result = consultation_state

    existing_traces = list(state.get("execution_traces") or [])
    subgraph_traces = list(graph_result.get("execution_traces") or [])
    execution_traces = list(existing_traces)
    for offset, raw_trace in enumerate(subgraph_traces, start=1):
        trace_payload = dict(raw_trace)
        trace_payload["sequence"] = len(existing_traces) + offset
        execution_traces.append(trace_payload)

    final_text = resolve_consultation_result_text(graph_result)
    token_usage = build_token_usage_from_execution_traces(execution_traces)

    return {
        "result": final_text,
        "messages": [AIMessage(content=final_text)] if final_text else [],
        "execution_traces": execution_traces,
        "token_usage": token_usage,
        "__interrupt__": graph_result.get("__interrupt__"),
    }


__all__ = [
    "consultation_agent",
    "has_pending_consultation_interrupt",
]
