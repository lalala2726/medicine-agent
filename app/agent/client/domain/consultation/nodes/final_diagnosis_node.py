from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.messages import SystemMessage

from app.agent.client.domain.consultation.helpers import (
    CONSULTATION_FINAL_DIAGNOSIS_MODEL_SLOT,
    append_trace_to_state,
    build_trace_item,
    resolve_natural_language_text,
)
from app.agent.client.domain.consultation.state import (
    CONSULTATION_STATUS_COMPLETED,
    ConsultationState,
)
from app.agent.client.domain.product.tools import (
    get_product_detail,
    get_product_spec,
    search_products,
)
from app.agent.client.domain.tools.card_tools import send_product_purchase_card
from app.core.agent.agent_event_bus import emit_answer_delta
from app.core.agent.agent_runtime import agent_stream
from app.core.agent.agent_tool_trace import record_agent_trace
from app.core.agent.base_prompt_middleware import BasePromptMiddleware
from app.core.config_sync import create_agent_chat_llm
from app.core.langsmith import traceable
from app.utils.prompt_utils import append_current_time_to_prompt, load_prompt

# consultation 最终诊断节点提示词。
CONSULTATION_FINAL_DIAGNOSIS_PROMPT = load_prompt("client/consultation/final_diagnosis_system_prompt.md")
# consultation 最终诊断节点兜底文本。
DEFAULT_FINAL_DIAGNOSIS_TEXT = "结合你目前提供的信息，更像是常见轻症方向；如果症状持续加重，请及时线下就医。"


@traceable(name="Client Consultation Final Diagnosis Node", run_type="chain")
def consultation_final_diagnosis_node(state: ConsultationState) -> dict[str, object]:
    """
    功能描述：
        给出最终诊断建议，并在需要时搜索商品、发送购买卡。

    参数说明：
        state (ConsultationState): 当前 consultation 状态。

    返回值：
        dict[str, object]: 当前节点写回的最终诊断结果。

    异常说明：
        无；节点执行异常由上层 workflow 统一处理。
    """

    history_messages = list(state.get("history_messages") or [])
    llm = create_agent_chat_llm(
        slot=CONSULTATION_FINAL_DIAGNOSIS_MODEL_SLOT,
        temperature=0.2,
        think=False,
    )
    llm_model_name = str(getattr(llm, "model_name", "") or "").strip() or "unknown"
    diagnosis_agent = create_agent(
        model=llm,
        tools=[
            search_products,
            get_product_detail,
            get_product_spec,
            send_product_purchase_card,
        ],
        system_prompt=SystemMessage(
            content=append_current_time_to_prompt(CONSULTATION_FINAL_DIAGNOSIS_PROMPT)
        ),
        middleware=[
            BasePromptMiddleware(base_prompt_file="client/_client_base_prompt.md"),
            ToolCallLimitMiddleware(thread_limit=5, run_limit=5),
        ],
    )
    stream_result = agent_stream(
        diagnosis_agent,
        history_messages,
        on_model_delta=emit_answer_delta,
    )
    trace_payload = record_agent_trace(
        payload=stream_result,
        input_messages=history_messages,
        fallback_text=str(stream_result.get("streamed_text") or ""),
    )
    final_text = resolve_natural_language_text(
        trace_text=str(trace_payload.get("text") or ""),
        fallback_text=str(stream_result.get("streamed_text") or ""),
        default_text=DEFAULT_FINAL_DIAGNOSIS_TEXT,
    )
    diagnosis_trace = build_trace_item(
        node_name="consultation_final_diagnosis_node",
        llm_model_name=llm_model_name or str(trace_payload.get("model_name") or "unknown"),
        output_text=final_text,
        llm_usage_complete=bool(trace_payload.get("is_usage_complete", False)),
        llm_token_usage=trace_payload.get("usage"),
        tool_calls=list(trace_payload.get("tool_calls") or []),
        node_context=None,
    )
    execution_traces, token_usage = append_trace_to_state(
        state=state,
        trace_item=diagnosis_trace,
    )
    return {
        "consultation_status": CONSULTATION_STATUS_COMPLETED,
        "diagnosis_ready": True,
        "comfort_text": "",
        "question_reply_text": "",
        "pending_question_text": "",
        "pending_question_options": [],
        "pending_ai_reply_text": "",
        "final_text": final_text,
        "diagnosis_trace": diagnosis_trace,
        "execution_traces": execution_traces,
        "token_usage": token_usage,
        "result": final_text,
        "messages": [],
    }


__all__ = [
    "CONSULTATION_FINAL_DIAGNOSIS_PROMPT",
    "consultation_final_diagnosis_node",
]
