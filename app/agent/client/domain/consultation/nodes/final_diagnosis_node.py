from __future__ import annotations

from typing import Any

from app.agent.client.domain.consultation.helpers import (
    DEFAULT_PURCHASE_QUANTITY,
    build_llm_agent,
    build_product_search_payload,
    build_recommendation_text,
    build_recommended_products,
    build_tool_trace,
    build_trace_item,
    collect_consultation_traces,
    invoke_runnable,
    resolve_final_result,
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
CONSULTATION_FINAL_PROMPT = load_prompt("client/consultation_final_diagnosis_system_prompt.md")


@traceable(name="Client Consultation Final Diagnosis Node", run_type="chain")
def consultation_final_diagnosis_node(state: ConsultationState) -> dict[str, Any]:
    """输出最终诊断文本，并在命中商品时发送购买确认卡。"""

    history_messages = list(state.get("history_messages") or [])
    agent, llm_model_name = build_llm_agent(
        state=state,
        prompt_text=CONSULTATION_FINAL_PROMPT,
    )
    result = agent_invoke(agent, history_messages)
    final_result = resolve_final_result(result.payload)

    tool_calls: list[dict[str, Any]] = []
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
            purchase_items = [
                {
                    "productId": product_id,
                    "quantity": DEFAULT_PURCHASE_QUANTITY,
                }
                for product_id in recommended_product_ids
            ]
            purchase_payload = {"items": purchase_items}
            invoke_runnable(send_product_purchase_card, purchase_payload)
            tool_calls.append(
                build_tool_trace(
                    tool_name="send_product_purchase_card",
                    tool_input=purchase_payload,
                )
            )

    trace_payload = record_agent_trace(
        payload=result.payload,
        input_messages=history_messages,
        fallback_text=result.content,
    )
    final_text = build_recommendation_text(
        diagnosis_text=final_result.diagnosis_text,
        product_names=product_names,
    )
    trace_item = build_trace_item(
        node_name="consultation_final_diagnosis_node",
        llm_model_name=llm_model_name,
        trace_payload=trace_payload,
        tool_calls=tool_calls,
        node_context={
            "recommended_product_ids": list(recommended_product_ids),
            "should_recommend_products": final_result.should_recommend_products,
        },
    )
    return {
        "consultation_status": CONSULTATION_STATUS_COMPLETED,
        "should_enter_diagnosis": True,
        "recommended_product_ids": recommended_product_ids,
        "final_text": final_text,
        "diagnosis_trace": trace_item,
        "node_traces": collect_consultation_traces(
            ConsultationState(
                **dict(state),
                diagnosis_trace=trace_item,
            )
        ),
    }


__all__ = [
    "CONSULTATION_FINAL_PROMPT",
    "consultation_final_diagnosis_node",
]
