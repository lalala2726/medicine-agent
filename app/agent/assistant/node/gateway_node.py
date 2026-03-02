from __future__ import annotations

import json
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage

from app.agent.assistant.state import AgentState, ExecutionTraceState, GatewayRoutingState
from app.core.agent.agent_runtime import agent_invoke
from app.core.agent.agent_tool_trace import record_agent_trace
from app.core.agent.base_prompt_middleware import BasePromptMiddleware
from app.core.langsmith import traceable
from app.core.llms.common import resolve_llm_value
from app.core.llms.provider import LlmProvider, resolve_provider
from app.core.llms import create_chat_model
from app.services.token_usage_service import append_trace_and_refresh_token_usage
from app.utils.prompt_utils import load_prompt

_GATEWAY_PROMPT = load_prompt("assistant/gateway_prompt.md")


def _resolve_gateway_router_model_name() -> str:
    """
    功能描述：
        解析 Gateway 路由节点专用模型名称。

    参数说明：
        无。

    返回值：
        str: Gateway 路由节点最终模型名称。

    异常说明：
        RuntimeError: 当 provider 对应的“专用模型配置 + 通用 chat 模型配置”均缺失时抛出。
    """

    resolved_provider = resolve_provider(None)
    if resolved_provider is LlmProvider.ALIYUN:
        dedicated_key = "DASHSCOPE_GATEWAY_ROUTER_MODEL"
        fallback_key = "DASHSCOPE_CHAT_MODEL"
    elif resolved_provider is LlmProvider.VOLCENGINE:
        dedicated_key = "VOLCENGINE_LLM_GATEWAY_ROUTER_MODEL"
        fallback_key = "VOLCENGINE_LLM_CHAT_MODEL"
    else:
        dedicated_key = "OPENAI_GATEWAY_ROUTER_MODEL"
        fallback_key = "OPENAI_CHAT_MODEL"

    gateway_model_name = resolve_llm_value(name=dedicated_key)
    if gateway_model_name:
        return gateway_model_name

    fallback_model_name = resolve_llm_value(name=fallback_key)
    if fallback_model_name:
        return fallback_model_name

    raise RuntimeError(f"{dedicated_key} and {fallback_key} are not set")


@traceable(name="Supervisor Gateway Router Node", run_type="chain")
def gateway_router(state: AgentState) -> dict[str, Any]:
    """
    功能描述：
        执行 Gateway 路由节点，输出路由目标与任务难度。

    参数说明：
        state (AgentState): LangGraph 节点状态，包含历史消息与执行追踪信息。

    返回值：
        dict[str, Any]: 节点输出状态增量，包含 `routing/execution_traces/token_usage`。

    异常说明：
        RuntimeError:
            - 当路由节点专用模型与通用模型均未配置时由 `_resolve_gateway_router_model_name` 抛出。
    """

    gateway_model_name = _resolve_gateway_router_model_name()
    llm = create_chat_model(
        model=gateway_model_name,
        temperature=0.0,
        # todo 适配不同模型之前的输出 JSON 格式
        extra_body={"response_format": {"type": "json_object"}},
        think=False,
    )
    agent = create_agent(
        model=llm,
        system_prompt=SystemMessage(content=_GATEWAY_PROMPT),
        middleware=[BasePromptMiddleware()],
    )
    history_messages = list(state.get("history_messages") or [])
    # 路由节点无需过多的上下文，只需要截取最近的20条消息
    result = agent_invoke(agent, history_messages[-20:])
    trace = record_agent_trace(
        payload=result.payload,
        input_messages=history_messages,
        fallback_text=result.content,
    )
    raw_content = trace.get("raw_content")

    parsed: dict[str, Any] = {}
    if isinstance(raw_content, dict):
        parsed = raw_content
    else:
        raw_text = str(trace.get("text") or "").strip()
        try:
            candidate = json.loads(raw_text)
            if isinstance(candidate, dict):
                parsed = candidate
        except json.JSONDecodeError:
            parsed = {}

    route_target = str(parsed.get("route_target") or "").strip()
    if route_target not in {"chat_agent", "supervisor_agent"}:
        route_target = "chat_agent"

    task_difficulty = str(parsed.get("task_difficulty") or "").strip().lower()
    if task_difficulty not in {"normal", "high"}:
        task_difficulty = "normal"

    routing: GatewayRoutingState = {
        "route_target": route_target,
        "task_difficulty": task_difficulty,
    }

    trace_item = ExecutionTraceState(
        node_name="gateway_router",
        model_name=gateway_model_name or str(trace.get("model_name") or "unknown"),
        output_text=str(trace.get("text") or ""),
        llm_used=True,
        llm_usage_complete=bool(trace.get("is_usage_complete", False)),
        llm_token_usage=trace.get("usage"),
        tool_calls=[],
    )
    execution_traces, token_usage = append_trace_and_refresh_token_usage(
        state.get("execution_traces"),
        trace_item,
    )
    return {
        "routing": routing,
        "execution_traces": execution_traces,
        "token_usage": token_usage,
    }
