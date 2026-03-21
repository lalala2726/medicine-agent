from __future__ import annotations

from typing import Any

from app.agent.client.domain.consultation.helpers import build_llm_agent, build_trace_item, resolve_status_result
from app.agent.client.domain.consultation.state import ConsultationState
from app.core.agent.agent_runtime import agent_invoke
from app.core.agent.agent_tool_trace import record_agent_trace
from app.core.langsmith import traceable
from app.utils.prompt_utils import load_prompt

# consultation 状态判断节点提示词。
CONSULTATION_STATUS_PROMPT = load_prompt("client/consultation_status_system_prompt.md")


@traceable(name="Client Consultation Status Node", run_type="chain")
def consultation_status_node(state: ConsultationState) -> dict[str, Any]:
    """判断当前是否已经满足进入最终诊断节点的条件。"""

    history_messages = list(state.get("history_messages") or [])
    agent, llm_model_name = build_llm_agent(
        state=state,
        prompt_text=CONSULTATION_STATUS_PROMPT,
        temperature=0.0,
    )
    result = agent_invoke(agent, history_messages)
    status_result = resolve_status_result(result.payload)
    trace_payload = record_agent_trace(
        payload=result.payload,
        input_messages=history_messages,
        fallback_text=result.content,
    )
    trace_item = build_trace_item(
        node_name="consultation_status_node",
        llm_model_name=llm_model_name,
        trace_payload=trace_payload,
        node_context={
            "consultation_status": status_result.consultation_status,
            "should_enter_diagnosis": status_result.should_enter_diagnosis,
        },
    )
    return {
        "consultation_status": status_result.consultation_status,
        "should_enter_diagnosis": status_result.should_enter_diagnosis,
        "status_trace": trace_item,
    }


__all__ = [
    "CONSULTATION_STATUS_PROMPT",
    "consultation_status_node",
]
