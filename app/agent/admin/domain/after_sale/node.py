from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.messages import AIMessage, SystemMessage

from app.agent.admin.domain.after_sale.tools import (
    get_admin_after_sale_detail,
    get_admin_after_sale_list,
)
from app.agent.admin.model_switch import model_switch
from app.agent.admin.state import AgentState, ExecutionTraceState
from app.core.agent.agent_event_bus import emit_answer_delta, emit_thinking_delta
from app.core.agent.agent_runtime import agent_stream
from app.core.agent.agent_tool_events import build_tool_status_middleware
from app.core.agent.agent_tool_trace import record_agent_trace
from app.core.agent.base_prompt_middleware import BasePromptMiddleware
from app.core.config_sync import create_agent_chat_llm
from app.core.langsmith import traceable
from app.services.token_usage_service import append_trace_and_refresh_token_usage
from app.utils.prompt_utils import load_prompt

_AFTER_SALE_NODE_SYSTEM_PROMPT = load_prompt("admin/after_sale_node_system_prompt.md")


@traceable(name="Admin Assistant After Sale Agent Node", run_type="chain")
def after_sale_agent(state: AgentState) -> dict[str, Any]:
    """
    功能描述：
        执行售后业务节点，处理售后列表筛选与售后详情查询任务。

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
    llm = create_agent_chat_llm(
        slot=model_switch(state),
        temperature=1.0,
        think=False,
    )
    llm_model_name = str(getattr(llm, "model_name", "") or "").strip()
    agent = create_agent(
        model=llm,
        system_prompt=SystemMessage(content=_AFTER_SALE_NODE_SYSTEM_PROMPT),
        tools=[
            get_admin_after_sale_list,
            get_admin_after_sale_detail,
        ],
        middleware=[
            BasePromptMiddleware(),
            build_tool_status_middleware(),
            ToolCallLimitMiddleware(thread_limit=5, run_limit=5),
        ],
    )
    stream_result = agent_stream(
        agent,
        history_messages,
        on_model_delta=emit_answer_delta,
        on_thinking_delta=emit_thinking_delta,
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
        node_name="after_sale_agent",
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
