"""
管理端单 Agent 节点实现。
"""

from __future__ import annotations

from typing import Any, Mapping, cast

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.messages import AIMessage, SystemMessage

from app.agent.admin.state import AgentState, ExecutionTraceState
from app.agent.admin.tools import (
    ADMIN_TOOL_REGISTRY,
    AdminDynamicToolMiddleware,
    bind_current_admin_tool_cache_conversation,
    render_admin_tool_cache_prompt,
    reset_current_admin_tool_cache_conversation,
)
from app.core.agent.agent_event_bus import emit_answer_delta, emit_thinking_delta
from app.core.agent.agent_runtime import agent_stream
from app.core.agent.agent_tool_events import build_tool_status_middleware
from app.core.agent.agent_tool_trace import record_agent_trace
from app.core.agent.base_prompt_middleware import BasePromptMiddleware
from app.core.agent.skill import SkillMiddleware
from app.core.config_sync import AgentChatModelSlot, create_agent_chat_llm
from app.core.langsmith import traceable
from app.services.token_usage_service import append_trace_and_refresh_token_usage
from app.utils.prompt_utils import append_current_time_to_prompt, load_prompt

# 单 admin agent 的系统提示词模板。
_ADMIN_AGENT_SYSTEM_PROMPT_TEMPLATE = SystemMessagePromptTemplate.from_template(
    load_prompt("admin/admin_agent_system_prompt.md"),
)
# admin agent 的状态 schema。
# 部分类型检查器不会把 `MessagesState` 子类识别为 `TypedDict` 类型，这里显式收窄以消除 IDE 误报。
_ADMIN_AGENT_STATE_SCHEMA = cast(Any, AgentState)


def _resolve_granted_tool_keys(stream_result: dict[str, Any]) -> list[str]:
    """
    功能描述：
        从单次 agent 流式执行结果中提取最终已授权工具数组。

    参数说明：
        stream_result (dict[str, Any]): `agent_stream` 返回结构。

    返回值：
        list[str]: 规范化后的已授权工具 key 数组。

    异常说明：
        无。
    """

    latest_state = stream_result.get("latest_state")
    if not isinstance(latest_state, Mapping):
        return []

    raw_granted_tool_keys = latest_state.get("granted_tool_keys")
    if not isinstance(raw_granted_tool_keys, list):
        return []

    granted_tool_keys: list[str] = []
    for raw_tool_key in raw_granted_tool_keys:
        tool_key = str(raw_tool_key or "").strip()
        if not tool_key:
            continue
        if tool_key in granted_tool_keys:
            continue
        granted_tool_keys.append(tool_key)
    return granted_tool_keys


@traceable(name="Admin Assistant Agent Node", run_type="chain")
def admin_agent(state: AgentState) -> dict[str, Any]:
    """
    功能描述：
        执行管理端单 Agent 节点，并在同轮内动态申请和注入业务工具。

    参数说明：
        state (AgentState): LangGraph 节点状态，主要读取历史消息与执行追踪。

    返回值：
        dict[str, Any]:
            节点输出状态增量，包含 `result/messages/execution_traces/token_usage/granted_tool_keys`。

    异常说明：
        不主动吞掉模型或工具异常；异常由上层流式主链路统一处理。
    """

    conversation_uuid = str(state.get("conversation_uuid") or "").strip()
    history_messages = list(state.get("history_messages") or [])
    admin_tool_cache_prompt = render_admin_tool_cache_prompt(conversation_uuid)
    admin_system_message = _ADMIN_AGENT_SYSTEM_PROMPT_TEMPLATE.format(
        tool_cache=admin_tool_cache_prompt,
    )
    llm = create_agent_chat_llm(
        slot=AgentChatModelSlot.ADMIN_BUSINESS_COMPLEX,
        temperature=1.0,
        think=False,
    )
    llm_model_name = str(getattr(llm, "model_name", "") or "").strip()
    agent = create_agent(
        model=llm,
        tools=ADMIN_TOOL_REGISTRY.all_tools,
        system_prompt=SystemMessage(
            content=append_current_time_to_prompt(
                str(admin_system_message.content or "")
            )
        ),
        state_schema=_ADMIN_AGENT_STATE_SCHEMA,
        middleware=[
            BasePromptMiddleware(),
            SkillMiddleware(),
            AdminDynamicToolMiddleware(registry=ADMIN_TOOL_REGISTRY),
            build_tool_status_middleware(),
            ToolCallLimitMiddleware(thread_limit=20, run_limit=10),
        ],
    )
    cache_token = bind_current_admin_tool_cache_conversation(conversation_uuid)
    try:
        stream_result = agent_stream(
            agent,
            history_messages,
            on_model_delta=emit_answer_delta,
            on_thinking_delta=emit_thinking_delta,
        )
    finally:
        reset_current_admin_tool_cache_conversation(cache_token)
    trace = record_agent_trace(
        payload=stream_result,
        input_messages=history_messages,
        fallback_text=str(stream_result.get("streamed_text") or ""),
    )
    current_execution_traces = list(state.get("execution_traces") or [])
    text = str(trace.get("text") or "").strip()
    trace_model_name = str(trace.get("model_name") or "").strip()
    granted_tool_keys = _resolve_granted_tool_keys(stream_result)
    trace_item = ExecutionTraceState(
        sequence=len(current_execution_traces) + 1,
        node_name="admin_agent",
        model_name=llm_model_name or trace_model_name or "unknown",
        status="success",
        output_text=text,
        llm_usage_complete=bool(trace.get("is_usage_complete", False)),
        llm_token_usage=trace.get("usage"),
        tool_calls=list(trace.get("tool_calls") or []),
        node_context={"granted_tool_keys": granted_tool_keys},
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
        "granted_tool_keys": granted_tool_keys,
    }


__all__ = [
    "admin_agent",
]
