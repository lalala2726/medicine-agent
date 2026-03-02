from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any
from typing import Literal

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field, ValidationError

from app.agent.assistant.state import AgentState, ExecutionTraceState
from app.core.agent.agent_runtime import agent_invoke
from app.core.agent.agent_tool_trace import record_agent_trace
from app.core.agent.base_prompt_middleware import BasePromptMiddleware
from app.core.langsmith import traceable
from app.core.llms import create_chat_model
from app.core.llms.common import resolve_llm_value
from app.core.llms.provider import LlmProvider, resolve_provider
from app.services.token_usage_service import append_trace_and_refresh_token_usage
from app.utils.prompt_utils import load_prompt

_GATEWAY_PROMPT = load_prompt("assistant/gateway_prompt.md")
_GATEWAY_ROUTE_FALLBACK: dict[str, Any] = {
    "route_targets": ["chat_agent"],
    "task_difficulty": "normal",
}
_ALLOWED_GATEWAY_TARGETS: tuple[str, ...] = (
    "chat_agent",
    "order_agent",
    "product_agent",
    "after_sale_agent",
    "user_agent",
    "analytics_agent",
)


class GatewayRoutingSchema(BaseModel):
    """
    功能描述：
        Gateway 路由节点结构化输出 Schema，约束路由目标数组与任务难度取值范围。

    参数说明：
        route_targets (list[Literal["chat_agent","order_agent","product_agent","after_sale_agent","user_agent","analytics_agent"]]):
            路由目标节点数组。
        task_difficulty (Literal["normal", "high"]): 任务难度等级。

    返回值：
        无（数据模型定义）。

    异常说明：
        pydantic.ValidationError: 当模型输出字段缺失或取值不合法时抛出。
    """

    route_targets: list[
        Literal[
            "chat_agent",
            "order_agent",
            "product_agent",
            "after_sale_agent",
            "user_agent",
            "analytics_agent",
        ]
    ] = Field(
        min_length=1,
        description=(
            "路由目标节点数组，仅允许 "
            "chat_agent/order_agent/product_agent/after_sale_agent/user_agent/analytics_agent"
        ),
    )
    task_difficulty: Literal["normal", "high"] = Field(
        description="任务难度，仅允许 normal 或 high",
    )


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


def _normalize_route_targets(route_targets: list[str]) -> list[str]:
    """
    功能描述：
        规范化网关路由目标数组并执行合法性校验。

    参数说明：
        route_targets (list[str]): 原始路由目标数组。

    返回值：
        list[str]:
            规范化后的路由目标数组（顺序去重）。
            当出现非法值、空值或 chat 与业务域混合时返回空数组。

    异常说明：
        无；解析失败由调用方决定兜底策略。
    """

    normalized: list[str] = []
    for raw_target in route_targets:
        target = str(raw_target or "").strip()
        if not target:
            return []
        if target not in _ALLOWED_GATEWAY_TARGETS:
            return []
        if target in normalized:
            continue
        normalized.append(target)

    if not normalized:
        return []
    if "chat_agent" in normalized and len(normalized) > 1:
        return []
    return normalized


def _resolve_gateway_routing_result(raw_payload: Any) -> dict[str, Any]:
    """
    功能描述：
        从 gateway 节点 agent 调用结果中解析结构化路由结果并做兜底校验。

    参数说明：
        raw_payload (Any): `agent_invoke(...).payload` 的原始结果对象。

    返回值：
        dict[str, Any]:
            合法路由结果字典，字段固定为：
            - `route_targets`：目标节点数组；
            - `task_difficulty`：任务难度。
            当结构化结果缺失或不合法时，回退 `chat_agent + normal`。

    异常说明：
        无；本函数内部对解析失败进行兜底，不向外抛出异常。
    """

    if not isinstance(raw_payload, Mapping):
        return dict(_GATEWAY_ROUTE_FALLBACK)
    structured_response = raw_payload.get("structured_response")

    structured_mapping: Mapping[str, Any] | None = None
    if isinstance(structured_response, Mapping):
        structured_mapping = structured_response
    elif isinstance(structured_response, BaseModel):
        structured_mapping = structured_response.model_dump()
    elif isinstance(structured_response, str):
        try:
            parsed_json = json.loads(structured_response)
        except json.JSONDecodeError:
            parsed_json = None
        if isinstance(parsed_json, Mapping):
            structured_mapping = parsed_json

    if structured_mapping is None:
        return dict(_GATEWAY_ROUTE_FALLBACK)
    try:
        parsed = GatewayRoutingSchema.model_validate(structured_mapping)
    except ValidationError:
        return dict(_GATEWAY_ROUTE_FALLBACK)
    normalized_targets = _normalize_route_targets(list(parsed.route_targets))
    if not normalized_targets:
        return dict(_GATEWAY_ROUTE_FALLBACK)
    return {
        "route_targets": normalized_targets,
        "task_difficulty": parsed.task_difficulty,
    }


@traceable(name="Assistant Gateway Router Node", run_type="chain")
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
        think=False,
    )
    agent = create_agent(
        model=llm,
        system_prompt=SystemMessage(content=_GATEWAY_PROMPT),
        middleware=[BasePromptMiddleware()],
        response_format=ToolStrategy(GatewayRoutingSchema),
    )
    history_messages = list(state.get("history_messages") or [])
    # 路由节点无需过多的上下文，只需要截取最近的20条消息
    result = agent_invoke(agent, history_messages[-20:])
    routing = _resolve_gateway_routing_result(result.payload)
    trace = record_agent_trace(
        payload=result.payload,
        input_messages=history_messages,
        fallback_text=result.content,
    )

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
