from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import SystemMessage

from app.agent.assistant.state import AgentState, ExecutionTraceState, GatewayRoutingState
from app.core.langsmith import traceable
from app.core.llm import create_chat_model
from app.services.token_usage_service import append_trace_and_refresh_token_usage
from app.utils.streaming_utils import (
    invoke_with_trace,
)

_GATEWAY_PROMPT = """
    你是药品商城后台助手的前置意图网关（Intent Gateway）。

    你只能输出 JSON，且必须包含两个字段：
    1) route_target: chat_agent|supervisor_agent
    2) task_difficulty: simple|normal|complex
    
    通用要求：
    1. 只输出一个 JSON 对象，不要输出任何解释、Markdown、代码块。
    2. 若用户输入很短（如“查询啊”），允许结合最近对话上下文延续同一业务域。
    3. 若上下文不足以判断业务域，优先路由 chat_agent，避免误调用业务节点。
    
    路由规则（优先级从高到低）：
    1. 业务相关，比如查询需要获取数据内的数据  -> supervisor_agent。
    4. 聊天或者是整理上一次对话内容并且不是没有获取数据的需求或者是已经有数据 -> chat_agent。
    
    难度规则：
    1. simple: 单步、参数明确、直接查询。
    2. normal: 需要少量推理或条件筛选。
    3. complex: 多阶段、负责、步骤超过3步。
    
    注意：一旦涉及 **supervisor_agent** 路由这边模型难度最低是 **normal**。
    
    示例：
    - 用户: "在吗"
      输出: {"route_target":"chat_agent","task_difficulty":"simple"}
    - 用户: "把上个月退款超过2次的订单找出来"
      输出: {"route_target":"supervisor_agent","task_difficulty":"complex"}
    """


@traceable(name="Supervisor Gateway Router Node", run_type="chain")
def gateway_router(state: AgentState) -> dict[str, Any]:
    llm = create_chat_model(
        model="qwen-flash",
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    history_messages = list(state.get("history_messages") or [])

    messages = [SystemMessage(content=_GATEWAY_PROMPT), *history_messages]

    trace = invoke_with_trace(llm, messages)
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
    if task_difficulty not in {"simple", "normal", "complex"}:
        task_difficulty = "normal"

    routing: GatewayRoutingState = {
        "route_target": route_target,
        "task_difficulty": task_difficulty,
    }

    trace_item = ExecutionTraceState(
        node_name="gateway_router",
        model_name=str(trace.get("model_name") or "unknown"),
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
