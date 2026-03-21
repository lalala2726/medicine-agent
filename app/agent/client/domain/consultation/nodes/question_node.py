from __future__ import annotations

from app.agent.client.domain.consultation.helpers import (
    CONSULTATION_QUESTION_MODEL_SLOT,
    DEFAULT_QUESTION_OPTIONS,
    DEFAULT_QUESTION_REPLY_TEXT,
    DEFAULT_QUESTION_TEXT,
    build_llm_agent,
    build_trace_item,
    resolve_question_result,
)
from app.agent.client.domain.consultation.state import (
    CONSULTATION_STATUS_COLLECTING,
    CONSULTATION_STATUS_COMPLETED,
    ConsultationState,
)
from app.core.agent.agent_runtime import agent_invoke
from app.core.agent.agent_tool_trace import record_agent_trace
from app.core.langsmith import traceable
from app.utils.prompt_utils import load_prompt

# consultation 追问节点提示词。
CONSULTATION_QUESTION_PROMPT = load_prompt("client/consultation/question_system_prompt.md")


@traceable(name="Client Consultation Question Node", run_type="chain")
def consultation_question_node(state: ConsultationState) -> dict[str, object]:
    """
    功能描述：
        以非流式结构化方式判断是否还需追问，并生成阶段性分析文本与问题卡片内容。

    参数说明：
        state (ConsultationState): 当前 consultation 状态。

    返回值：
        dict[str, object]: 当前节点写回的追问判断结果。

    异常说明：
        无；节点执行异常由上层 workflow 统一处理。
    """

    history_messages = list(state.get("history_messages") or [])
    agent, llm_model_name = build_llm_agent(
        state=state,
        slot=CONSULTATION_QUESTION_MODEL_SLOT,
        prompt_text=CONSULTATION_QUESTION_PROMPT,
    )
    result = agent_invoke(agent, history_messages)
    question_result = resolve_question_result(result.payload)
    trace_payload = record_agent_trace(
        payload=result.payload,
        input_messages=history_messages,
        fallback_text=result.content,
    )

    if question_result.diagnosis_ready:
        trace_output_text = "当前信息已足够进入最终诊断。"
        question_trace = build_trace_item(
            node_name="consultation_question_node",
            llm_model_name=llm_model_name or str(trace_payload.get("model_name") or "unknown"),
            output_text=trace_output_text,
            llm_usage_complete=bool(trace_payload.get("is_usage_complete", False)),
            llm_token_usage=trace_payload.get("usage"),
            tool_calls=list(trace_payload.get("tool_calls") or []),
            node_context={
                "diagnosis_ready": True,
            },
        )
        return {
            "consultation_status": CONSULTATION_STATUS_COMPLETED,
            "diagnosis_ready": True,
            "question_reply_text": "",
            "pending_question_text": "",
            "pending_question_options": [],
            "question_trace": question_trace,
        }

    question_reply_text = question_result.question_reply_text or DEFAULT_QUESTION_REPLY_TEXT
    question_text = question_result.question_text or DEFAULT_QUESTION_TEXT
    options = list(question_result.options or DEFAULT_QUESTION_OPTIONS)
    question_trace = build_trace_item(
        node_name="consultation_question_node",
        llm_model_name=llm_model_name or str(trace_payload.get("model_name") or "unknown"),
        output_text=question_reply_text,
        llm_usage_complete=bool(trace_payload.get("is_usage_complete", False)),
        llm_token_usage=trace_payload.get("usage"),
        tool_calls=list(trace_payload.get("tool_calls") or []),
        node_context={
            "diagnosis_ready": False,
            "question_text": question_text,
            "options": options,
        },
    )
    return {
        "consultation_status": CONSULTATION_STATUS_COLLECTING,
        "diagnosis_ready": False,
        "question_reply_text": question_reply_text,
        "pending_question_text": question_text,
        "pending_question_options": options,
        "question_trace": question_trace,
    }


__all__ = [
    "CONSULTATION_QUESTION_PROMPT",
    "consultation_question_node",
]
