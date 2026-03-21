from __future__ import annotations

from typing import Any

from app.agent.client.domain.consultation.helpers import DEFAULT_COMFORT_TEXT, build_llm_agent, build_trace_item
from app.agent.client.domain.consultation.state import ConsultationState
from app.core.agent.agent_runtime import agent_invoke
from app.core.agent.agent_tool_trace import record_agent_trace
from app.core.langsmith import traceable
from app.utils.prompt_utils import load_prompt

# consultation 安抚说明节点提示词。
CONSULTATION_COMFORT_PROMPT = load_prompt("client/consultation_comfort_system_prompt.md")


@traceable(name="Client Consultation Comfort Node", run_type="chain")
def consultation_comfort_node(state: ConsultationState) -> dict[str, Any]:
    """生成病情咨询的安抚说明文本。"""

    history_messages = list(state.get("history_messages") or [])
    agent, llm_model_name = build_llm_agent(
        state=state,
        prompt_text=CONSULTATION_COMFORT_PROMPT,
    )
    result = agent_invoke(agent, history_messages)
    trace_payload = record_agent_trace(
        payload=result.payload,
        input_messages=history_messages,
        fallback_text=result.content,
    )
    comfort_text = str(trace_payload.get("text") or "").strip() or DEFAULT_COMFORT_TEXT
    trace_item = build_trace_item(
        node_name="consultation_comfort_node",
        llm_model_name=llm_model_name,
        trace_payload=trace_payload,
        node_context=None,
    )
    return {
        "comfort_text": comfort_text,
        "comfort_trace": trace_item,
    }


__all__ = [
    "CONSULTATION_COMFORT_PROMPT",
    "consultation_comfort_node",
]
