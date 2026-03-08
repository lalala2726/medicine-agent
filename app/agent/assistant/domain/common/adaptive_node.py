from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.messages import AIMessage, SystemMessage

from app.agent.assistant.model_switch import model_switch
from app.agent.assistant.state import AgentState, ExecutionTraceState
from app.agent.assistant.domain.after_sale.tools import (
    get_admin_after_sale_detail,
    get_admin_after_sale_list,
)
from app.agent.assistant.domain.analytics.tools import (
    get_analytics_hot_products,
    get_analytics_order_status_distribution,
    get_analytics_order_trend,
    get_analytics_overview,
    get_analytics_payment_distribution,
    get_analytics_product_return_rates,
)
from app.agent.assistant.domain.common.tools import get_current_time
from app.agent.assistant.domain.common.tools import get_safe_user_info
from app.agent.assistant.domain.order.tools import (
    get_order_list,
    get_order_shipping,
    get_order_timeline,
    get_orders_detail,
)
from app.agent.assistant.domain.product.tools import (
    get_drug_detail,
    get_product_detail,
    get_product_list,
)
from app.agent.assistant.domain.user.tools import (
    get_admin_user_consume_info,
    get_admin_user_detail,
    get_admin_user_list,
    get_admin_user_wallet,
    get_admin_user_wallet_flow,
)
from app.core.agent.agent_event_bus import emit_answer_delta, emit_thinking_delta
from app.core.agent.agent_runtime import agent_stream
from app.core.agent.agent_tool_events import build_tool_status_middleware
from app.core.agent.agent_tool_trace import record_agent_trace
from app.core.agent.base_prompt_middleware import BasePromptMiddleware
from app.core.langsmith import traceable
from app.core.llms import create_chat_model
from app.core.agent.skill import SkillMiddleware
from app.services.token_usage_service import append_trace_and_refresh_token_usage
from app.utils.prompt_utils import load_prompt

_ADAPTIVE_AGENT_SYSTEM_PROMPT = load_prompt("assistant/adaptive_agent_system_prompt.md")

_DOMAIN_TOOL_MAP: dict[str, tuple[Any, ...]] = {
    "order_agent": (
        get_order_list,
        get_orders_detail,
        get_order_timeline,
        get_order_shipping,
    ),
    "product_agent": (
        get_product_list,
        get_product_detail,
        get_drug_detail,
    ),
    "after_sale_agent": (
        get_admin_after_sale_list,
        get_admin_after_sale_detail,
    ),
    "user_agent": (
        get_admin_user_list,
        get_admin_user_detail,
        get_admin_user_wallet,
        get_admin_user_wallet_flow,
        get_admin_user_consume_info,
    ),
    "analytics_agent": (
        get_analytics_overview,
        get_analytics_order_trend,
        get_analytics_order_status_distribution,
        get_analytics_payment_distribution,
        get_analytics_hot_products,
        get_analytics_product_return_rates,
    ),
}
_BASE_ADAPTIVE_TOOLS: tuple[Any, ...] = (
    get_current_time,
    get_safe_user_info,
)


def _resolve_adaptive_route_targets(state: AgentState) -> list[str]:
    """
    功能描述：
        从节点状态中解析并规范化 adaptive 节点可用的业务域路由目标。

    参数说明：
        state (AgentState): LangGraph 节点状态，预期包含 `routing.route_targets`。

    返回值：
        list[str]:
            规范化后的业务域目标列表，仅保留 `_DOMAIN_TOOL_MAP` 支持的目标，
            且按出现顺序去重。

    异常说明：
        无；输入结构异常时返回空列表。
    """

    routing = state.get("routing")
    if not isinstance(routing, dict):
        return []
    raw_targets = routing.get("route_targets")
    if not isinstance(raw_targets, list):
        return []

    normalized_targets: list[str] = []
    for raw_target in raw_targets:
        target = str(raw_target or "").strip()
        if not target:
            continue
        if target not in _DOMAIN_TOOL_MAP:
            continue
        if target in normalized_targets:
            continue
        normalized_targets.append(target)
    return normalized_targets


def _build_adaptive_tools(route_targets: list[str]) -> list[Any]:
    """
    功能描述：
        基于网关输出的业务域列表动态构建 adaptive 节点工具集合。

    参数说明：
        route_targets (list[str]): 已规范化的业务域目标列表。

    返回值：
        list[Any]:
            动态工具集合，规则为：
            1. 先按 `route_targets` 顺序追加对应业务域工具；
            2. 再追加基础工具（当前时间、用户信息）；
            3. 全过程按对象标识去重。

    异常说明：
        无；即使 `route_targets` 为空，也会返回基础工具集合。
    """

    dynamic_tools: list[Any] = []
    seen_tool_ids: set[int] = set()

    def _append_tool(tool_obj: Any) -> None:
        """
        功能描述：
            将单个工具按“未出现才追加”规则加入动态工具集合。

        参数说明：
            tool_obj (Any): 工具对象。

        返回值：
            None。

        异常说明：
            无。
        """

        tool_identity = id(tool_obj)
        if tool_identity in seen_tool_ids:
            return
        seen_tool_ids.add(tool_identity)
        dynamic_tools.append(tool_obj)

    for target in route_targets:
        for tool_obj in _DOMAIN_TOOL_MAP.get(target, ()):
            _append_tool(tool_obj)
    for base_tool in _BASE_ADAPTIVE_TOOLS:
        _append_tool(base_tool)
    return dynamic_tools


@traceable(name="Assistant Adaptive Agent Node", run_type="chain")
def adaptive_agent(state: AgentState) -> dict[str, Any]:
    """
    功能描述：
        执行自适应业务节点，处理多业务域联动任务，并根据网关路由结果动态装配工具集。

    参数说明：
        state (AgentState): LangGraph 节点状态；主要读取 `routing/history_messages/execution_traces`。

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

    model_name, enable_think = model_switch(state)
    adaptive_route_targets = _resolve_adaptive_route_targets(state)
    adaptive_tools = _build_adaptive_tools(adaptive_route_targets)
    history_messages = list(state.get("history_messages") or [])
    llm = create_chat_model(
        model=model_name,
        temperature=1.0,
        think=enable_think,
    )
    llm_model_name = str(getattr(llm, "model_name", "") or "").strip()
    agent = create_agent(
        model=llm,
        system_prompt=SystemMessage(content=_ADAPTIVE_AGENT_SYSTEM_PROMPT),
        tools=adaptive_tools,
        middleware=[
            BasePromptMiddleware(),
            build_tool_status_middleware(),
            SkillMiddleware(scope="adaptive"),
            ToolCallLimitMiddleware(thread_limit=20, run_limit=10),
        ],
    )
    stream_result = agent_stream(
        agent,
        history_messages,
        on_model_delta=emit_answer_delta,
        on_thinking_delta=emit_thinking_delta if enable_think else None,
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
        node_name="adaptive_agent",
        model_name=llm_model_name or trace_model_name or model_name or "unknown",
        status="success",
        output_text=text,
        llm_usage_complete=bool(trace.get("is_usage_complete", False)),
        llm_token_usage=trace.get("usage"),
        tool_calls=list(trace.get("tool_calls") or []),
        node_context={"route_targets": adaptive_route_targets},
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
