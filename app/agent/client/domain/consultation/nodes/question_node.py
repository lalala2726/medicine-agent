from __future__ import annotations

from typing import Any

from app.agent.client.domain.consultation.helpers import (
    build_llm_agent,
    build_tool_trace,
    build_trace_item,
    invoke_runnable,
    resolve_question_result,
)
from app.agent.client.domain.consultation.state import ConsultationState
from app.agent.client.domain.tools.card_tools import send_selection_card
from app.core.agent.agent_runtime import agent_invoke
from app.core.agent.agent_tool_trace import record_agent_trace
from app.core.langsmith import traceable
from app.utils.prompt_utils import load_prompt

# consultation 问询卡片节点提示词。
CONSULTATION_QUESTION_PROMPT = load_prompt("client/consultation_question_system_prompt.md")


@traceable(name="Client Consultation Question Card Node", run_type="chain")
def consultation_question_card_node(state: ConsultationState) -> dict[str, Any]:
    """生成病情咨询追问文本，并按需发送选择卡片。"""

    history_messages = list(state.get("history_messages") or [])
    agent, llm_model_name = build_llm_agent(
        state=state,
        prompt_text=CONSULTATION_QUESTION_PROMPT,
    )
    result = agent_invoke(agent, history_messages)
    question_result = resolve_question_result(result.payload)

    tool_calls: list[dict[str, Any]] = []
    selection_card_result: Any = None
    if not question_result.should_enter_diagnosis:
        selection_card_payload = {
            "title": question_result.question_text,
            "options": list(question_result.options),
        }
        selection_card_result = invoke_runnable(
            send_selection_card,
            selection_card_payload,
        )
        tool_calls.append(
            build_tool_trace(
                tool_name="send_selection_card",
                tool_input=selection_card_payload,
            )
        )

    trace_payload = record_agent_trace(
        payload=result.payload,
        input_messages=history_messages,
        fallback_text=result.content,
    )
    trace_item = build_trace_item(
        node_name="consultation_question_card_node",
        llm_model_name=llm_model_name,
        trace_payload=trace_payload,
        tool_calls=tool_calls,
        node_context={
            "consultation_status": question_result.consultation_status,
            "should_enter_diagnosis": question_result.should_enter_diagnosis,
            "selection_card_result": selection_card_result,
        },
    )
    return {
        "consultation_status": question_result.consultation_status,
        "should_enter_diagnosis": question_result.should_enter_diagnosis,
        "question_text": (
            ""
            if question_result.should_enter_diagnosis
            else question_result.question_text
        ),
        "question_trace": trace_item,
    }


__all__ = [
    "CONSULTATION_QUESTION_PROMPT",
    "consultation_question_card_node",
]
