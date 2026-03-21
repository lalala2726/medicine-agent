from __future__ import annotations

from app.agent.client.domain.consultation.helpers import (
    CONSULTATION_COMFORT_MODEL_SLOT,
    DEFAULT_COMFORT_TEXT,
    build_llm_agent,
    build_trace_item,
    resolve_natural_language_text,
)
from app.agent.client.domain.consultation.state import ConsultationState
from app.core.agent.agent_event_bus import emit_answer_delta
from app.core.agent.agent_runtime import agent_stream
from app.core.agent.agent_tool_trace import record_agent_trace
from app.core.langsmith import traceable
from app.utils.prompt_utils import load_prompt

# consultation 安抚节点提示词。
CONSULTATION_COMFORT_PROMPT = load_prompt("client/consultation/comfort_system_prompt.md")


@traceable(name="Client Consultation Comfort Node", run_type="chain")
def consultation_comfort_node(state: ConsultationState) -> dict[str, object]:
    """
    功能描述：
        在 collecting 阶段先流式输出安抚与基于已知信息的初步说明。

    参数说明：
        state (ConsultationState): 当前 consultation 状态。

    返回值：
        dict[str, object]: 当前节点写回的 comfort 文本与 trace。

    异常说明：
        无；节点执行异常由上层 workflow 统一处理。
    """

    history_messages = list(state.get("history_messages") or [])
    agent, llm_model_name = build_llm_agent(
        state=state,
        slot=CONSULTATION_COMFORT_MODEL_SLOT,
        temperature=1.3,
        prompt_text=CONSULTATION_COMFORT_PROMPT,
    )
    stream_result = agent_stream(
        agent,
        history_messages,
        on_model_delta=emit_answer_delta,
    )
    trace_payload = record_agent_trace(
        payload=stream_result,
        input_messages=history_messages,
        fallback_text=str(stream_result.get("streamed_text") or ""),
    )
    comfort_text = resolve_natural_language_text(
        trace_text=str(trace_payload.get("text") or ""),
        fallback_text=str(stream_result.get("streamed_text") or ""),
        default_text=DEFAULT_COMFORT_TEXT,
    )
    comfort_trace = build_trace_item(
        node_name="consultation_comfort_node",
        llm_model_name=llm_model_name or str(trace_payload.get("model_name") or "unknown"),
        output_text=comfort_text,
        llm_usage_complete=bool(trace_payload.get("is_usage_complete", False)),
        llm_token_usage=trace_payload.get("usage"),
        tool_calls=list(trace_payload.get("tool_calls") or []),
        node_context=None,
    )
    return {
        "comfort_text": comfort_text,
        "comfort_trace": comfort_trace,
    }


__all__ = [
    "CONSULTATION_COMFORT_PROMPT",
    "consultation_comfort_node",
]
