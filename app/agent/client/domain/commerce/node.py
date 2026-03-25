"""
客户端 commerce 复合节点实现。
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts import SystemMessagePromptTemplate

from app.agent.client.domain.after_sale.tools import (
    check_after_sale_eligibility,
    get_after_sale_detail,
)
from app.agent.client.domain.order.tools import (
    check_order_cancelable,
    get_order_detail,
    get_order_shipping,
    get_order_timeline,
)
from app.agent.client.domain.product.tools import (
    get_product_detail,
    get_product_spec,
    search_products,
)
from app.agent.client.domain.tools.action_tools import (
    open_user_after_sale_list,
    open_user_order_list,
)
from app.agent.client.state import AgentState, ExecutionTraceState
from app.core.agent.agent_event_bus import emit_answer_delta, emit_thinking_delta
from app.core.agent.agent_runtime import agent_stream
from app.core.agent.agent_tool_trace import record_agent_trace
from app.core.agent.base_prompt_middleware import BasePromptMiddleware
from app.core.agent.skill import SkillMiddleware
from app.core.agent.tool_cache import (
    CLIENT_COMMERCE_TOOL_CACHE_PROFILE,
    bind_tool_cache_conversation,
    render_tool_cache_prompt,
    reset_tool_cache_conversation,
)
from app.core.config_sync import AgentChatModelSlot, create_agent_chat_llm
from app.core.langsmith import traceable
from app.services.token_usage_service import append_trace_and_refresh_token_usage
from app.utils.prompt_utils import append_current_time_to_prompt, load_prompt

# commerce 节点系统提示词模板。
_COMMERCE_SYSTEM_PROMPT_TEMPLATE = SystemMessagePromptTemplate.from_template(
    load_prompt("client/commerce_node_system_prompt.md"),
)
# commerce 节点单轮最多允许的工具调用次数。
_COMMERCE_TOOL_CALL_THREAD_LIMIT = 10
# commerce 节点单次运行最多允许的工具调用次数。
_COMMERCE_TOOL_CALL_RUN_LIMIT = 8


@traceable(name="Client Assistant Commerce Agent Node", run_type="chain")
def commerce_agent(state: AgentState) -> dict[str, Any]:
    """
    功能描述：
        执行 client commerce 复合节点，统一处理商品、订单与售后问题。

    参数说明：
        state (AgentState): 当前 client agent 工作流状态。

    返回值：
        dict[str, Any]:
            commerce 节点输出结果，包含回答文本、消息、执行轨迹与 token 汇总。

    异常说明：
        不主动吞掉模型或工具异常；异常由上层工作流统一处理。
    """

    conversation_uuid = str(state.get("conversation_uuid") or "").strip()
    history_messages = list(state.get("history_messages") or [])
    commerce_tool_cache_prompt = render_tool_cache_prompt(
        CLIENT_COMMERCE_TOOL_CACHE_PROFILE,
        conversation_uuid,
    )
    commerce_system_message = _COMMERCE_SYSTEM_PROMPT_TEMPLATE.format(
        tool_cache=commerce_tool_cache_prompt,
    )
    llm = create_agent_chat_llm(
        slot=AgentChatModelSlot.CLIENT_CHAT,
        temperature=1.0,
        think=False,
    )
    llm_model_name = str(getattr(llm, "model_name", "") or "").strip()
    agent = create_agent(
        model=llm,
        tools=[
            open_user_order_list,
            get_order_detail,
            get_order_shipping,
            get_order_timeline,
            check_order_cancelable,
            search_products,
            get_product_detail,
            get_product_spec,
            open_user_after_sale_list,
            get_after_sale_detail,
            check_after_sale_eligibility,
        ],
        system_prompt=SystemMessage(
            content=append_current_time_to_prompt(
                str(commerce_system_message.content or "")
            )
        ),
        middleware=[
            BasePromptMiddleware(base_prompt_file="client/_client_base_prompt.md"),
            SkillMiddleware(skill_scope="client_commerce"),
            ToolCallLimitMiddleware(
                thread_limit=_COMMERCE_TOOL_CALL_THREAD_LIMIT,
                run_limit=_COMMERCE_TOOL_CALL_RUN_LIMIT,
            ),
        ],
    )
    cache_token = bind_tool_cache_conversation(
        CLIENT_COMMERCE_TOOL_CACHE_PROFILE,
        conversation_uuid,
    )
    try:
        stream_result = agent_stream(
            agent,
            history_messages,
            on_model_delta=emit_answer_delta,
            on_thinking_delta=emit_thinking_delta,
        )
    finally:
        reset_tool_cache_conversation(
            CLIENT_COMMERCE_TOOL_CACHE_PROFILE,
            cache_token,
        )
    trace = record_agent_trace(
        payload=stream_result,
        input_messages=history_messages,
        fallback_text=str(stream_result.get("streamed_text") or ""),
    )
    current_execution_traces = list(state.get("execution_traces") or [])
    text = str(trace.get("text") or "").strip()
    trace_model_name = str(trace.get("model_name") or "").strip()
    trace_item = ExecutionTraceState(
        sequence=len(current_execution_traces) + 1,
        node_name="commerce_agent",
        model_name=llm_model_name or trace_model_name or "unknown",
        status="success",
        output_text=text,
        llm_usage_complete=bool(trace.get("is_usage_complete", False)),
        llm_token_usage=trace.get("usage"),
        tool_calls=list(trace.get("tool_calls") or []),
        node_context=None,
    )
    execution_traces, token_usage = append_trace_and_refresh_token_usage(
        current_execution_traces,
        trace_item,
    )
    return {
        "result": text,
        "messages": [AIMessage(content=text)],
        "execution_traces": execution_traces,
        "token_usage": token_usage,
    }
