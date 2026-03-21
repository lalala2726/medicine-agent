from __future__ import annotations

from app.agent.client.domain.consultation.helpers import (
    DEFAULT_PURCHASE_QUANTITY,
    append_trace_to_state,
    build_llm_agent,
    build_product_search_payload,
    build_recommendation_text,
    build_recommended_products,
    build_tool_trace,
    build_trace_item,
    emit_consultation_answer_deltas_async,
    invoke_runnable,
    resolve_final_diagnosis_result,
    run_async_safely,
)
from app.agent.client.domain.consultation.state import (
    CONSULTATION_STATUS_COMPLETED,
    ConsultationState,
)
from app.agent.client.domain.product.tools import search_products
from app.agent.client.domain.tools.card_tools import send_product_purchase_card
from app.core.agent.agent_runtime import agent_invoke
from app.core.agent.agent_tool_trace import record_agent_trace
from app.core.langsmith import traceable
from app.utils.prompt_utils import load_prompt

# consultation 最终诊断节点提示词。
CONSULTATION_FINAL_DIAGNOSIS_PROMPT = load_prompt("client/consultation_final_diagnosis_system_prompt.md")


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
    agent, llm_model_name = build_llm_agent(
        state=state,
        prompt_text=CONSULTATION_FINAL_DIAGNOSIS_PROMPT,
    )
    result = agent_invoke(agent, history_messages)
    final_result = resolve_final_diagnosis_result(result.payload)

    tool_calls: list[dict[str, object]] = []
    recommended_product_ids: list[int] = []
    product_names: list[str] = []

    search_payload = build_product_search_payload(final_result=final_result)
    if final_result.should_recommend_products and search_payload is not None:
        search_result = invoke_runnable(search_products, search_payload)
        tool_calls.append(
            build_tool_trace(
                tool_name="search_products",
                tool_input=search_payload,
            )
        )
        recommended_product_ids, product_names = build_recommended_products(search_result)

        if recommended_product_ids:
            purchase_payload = {
                "items": [
                    {
                        "productId": product_id,
                        "quantity": DEFAULT_PURCHASE_QUANTITY,
                    }
                    for product_id in recommended_product_ids
                ]
            }
            invoke_runnable(send_product_purchase_card, purchase_payload)
            tool_calls.append(
                build_tool_trace(
                    tool_name="send_product_purchase_card",
                    tool_input=purchase_payload,
                )
            )

    final_text = build_recommendation_text(
        diagnosis_text=final_result.diagnosis_text,
        product_names=product_names,
    )
    run_async_safely(emit_consultation_answer_deltas_async(final_text))

    trace_payload = record_agent_trace(
        payload=result.payload,
        input_messages=history_messages,
        fallback_text=result.content,
    )
    diagnosis_trace = build_trace_item(
        node_name="consultation_final_diagnosis_node",
        llm_model_name=llm_model_name or str(trace_payload.get("model_name") or "unknown"),
        output_text=final_text,
        llm_usage_complete=bool(trace_payload.get("is_usage_complete", False)),
        llm_token_usage=trace_payload.get("usage"),
        tool_calls=tool_calls,
        node_context={
            "should_recommend_products": final_result.should_recommend_products,
            "recommended_product_ids": list(recommended_product_ids),
            "product_keyword": final_result.product_keyword,
            "product_usage": final_result.product_usage,
        },
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
        "recommended_product_ids": recommended_product_ids,
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
