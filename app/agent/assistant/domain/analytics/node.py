from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.messages import AIMessage, SystemMessage

from app.agent.assistant.state import AgentState, ExecutionTraceState
from app.agent.assistant.domain.analytics.tools import (
    get_analytics_hot_products,
    get_analytics_order_status_distribution,
    get_analytics_order_trend,
    get_analytics_overview,
    get_analytics_payment_distribution,
    get_analytics_product_return_rates,
)
from app.core.agent.agent_event_bus import emit_answer_delta
from app.core.agent.agent_runtime import agent_stream
from app.core.agent.agent_tool_events import build_tool_status_middleware
from app.core.agent.agent_tool_trace import record_agent_trace
from app.core.agent.base_prompt_middleware import BasePromptMiddleware
from app.core.langsmith import traceable
from app.core.llms import create_chat_model
from app.core.agent.skill import SkillMiddleware
from app.services.token_usage_service import append_trace_and_refresh_token_usage
from app.utils.prompt_utils import load_prompt

_ANALYTICS_NODE_SYSTEM_PROMPT = load_prompt("assistant/analytics_node_system_prompt.md")


@traceable(name="Assistant Analytics Agent Node", run_type="chain")
def analytics_agent(state: AgentState) -> dict[str, Any]:
    """
    功能描述：
        执行运营分析业务节点，处理运营总览、趋势、分布与排行等分析任务。

    参数说明：
        state (AgentState): LangGraph 节点状态；主要读取 `history_messages` 与 `execution_traces`。

    返回值：
        dict[str, Any]:
            返回节点状态增量，包含：
            - `result` (str): 节点最终输出文本；
            - `messages` (list[AIMessage]): 供后续状态消费的 AI 消息；
            - `execution_traces` (list[ExecutionTraceState]): 追加后的执行追踪；
            - `token_usage` (dict | None): 刷新后的消息级 token 汇总。

    异常说明：
        不主动抛出业务异常；模型、工具与中间件链路异常由上层统一捕获与降级处理。
    """

    history_messages = list(state.get("history_messages") or [])
    llm = create_chat_model(
        temperature=1.0,
        think=False,
    )
    llm_model_name = str(getattr(llm, "model_name", "") or "").strip()
    agent = create_agent(
        model=llm,
        system_prompt=SystemMessage(content=_ANALYTICS_NODE_SYSTEM_PROMPT),
        tools=[
            get_analytics_overview,
            get_analytics_order_trend,
            get_analytics_order_status_distribution,
            get_analytics_payment_distribution,
            get_analytics_hot_products,
            get_analytics_product_return_rates,
        ],
        middleware=[
            BasePromptMiddleware(),
            build_tool_status_middleware(),
            SkillMiddleware(skill_scope="chart"),
            ToolCallLimitMiddleware(thread_limit=5, run_limit=5),
        ],
    )
    stream_result = agent_stream(
        agent,
        history_messages,
        on_model_delta=emit_answer_delta,
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
        node_name="analytics_agent",
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
