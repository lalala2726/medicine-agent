from __future__ import annotations

from app.agent.client.domain.consultation.helpers import (
    CONSULTATION_ROUTE_MODEL_SLOT,
    CONSULTATION_ROUTE_TEMPERATURE,
    append_trace_to_state,
    build_consultation_input_messages,
    build_llm_agent,
    build_trace_item,
    resolve_route_result,
)
from app.agent.client.domain.consultation.state import ConsultationState
from app.core.agent.agent_runtime import agent_invoke
from app.core.agent.agent_tool_trace import record_agent_trace
from app.core.langsmith import traceable
from app.utils.prompt_utils import load_prompt

# consultation 子图内路由节点提示词。
CONSULTATION_ROUTE_PROMPT = load_prompt("client/consultation/route_system_prompt.md")


@traceable(name="Client Consultation Route Node", run_type="chain")
def consultation_route_node(state: ConsultationState) -> dict[str, object]:
    """
    功能描述：
        在 consultation 子图入口执行非流式结构化路由，判断当前问题应直接回答、继续追问或进入最终诊断。

    参数说明：
        state (ConsultationState): 当前 consultation 状态。

    返回值：
        dict[str, object]: 路由节点写回的结构化路由结果与 trace 信息。

    异常说明：
        无；节点执行异常由上层 workflow 统一处理。
    """

    input_messages = build_consultation_input_messages(
        state=state,
        include_progress_context=True,
    )
    agent, llm_model_name = build_llm_agent(
        state=state,
        slot=CONSULTATION_ROUTE_MODEL_SLOT,
        temperature=CONSULTATION_ROUTE_TEMPERATURE,
        prompt_text=CONSULTATION_ROUTE_PROMPT,
    )
    result = agent_invoke(agent, input_messages)
    route_result = resolve_route_result(result.payload)
    trace_payload = record_agent_trace(
        payload=result.payload,
        input_messages=input_messages,
        fallback_text=result.content,
    )
    route_trace = build_trace_item(
        node_name="consultation_route_node",
        llm_model_name=llm_model_name or str(trace_payload.get("model_name") or "unknown"),
        output_text=route_result.reason,
        llm_usage_complete=bool(trace_payload.get("is_usage_complete", False)),
        llm_token_usage=trace_payload.get("usage"),
        tool_calls=list(trace_payload.get("tool_calls") or []),
        node_context={
            "next_action": route_result.next_action,
            "consultation_mode": route_result.consultation_mode,
        },
    )
    execution_traces, token_usage = append_trace_to_state(
        state=state,
        trace_item=route_trace,
    )
    return {
        "consultation_route": {
            "next_action": route_result.next_action,
            "consultation_mode": route_result.consultation_mode,
            "reason": route_result.reason,
        },
        "diagnosis_ready": route_result.next_action == "final_diagnosis",
        "route_trace": route_trace,
        "execution_traces": execution_traces,
        "token_usage": token_usage,
        "result": "",
        "messages": [],
    }


__all__ = [
    "CONSULTATION_ROUTE_PROMPT",
    "consultation_route_node",
]
