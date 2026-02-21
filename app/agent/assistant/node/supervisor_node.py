from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, SystemMessage

from app.agent.assistant.state import AgentState, ExecutionTraceState
from app.agent.assistant.tools.order_tool import order_tool_agent
from app.agent.assistant.tools.product_tool import product_tool_agent
from app.core.langsmith import traceable
from app.core.llm import create_chat_model
from app.schemas.prompt import base_prompt
from app.services.token_usage_service import append_trace_and_refresh_token_usage
from app.utils.streaming_utils import invoke_with_trace

_SUPERVISOR_PROMPT = """
    你是药品商城后台管理助手的 supervisor 节点。
    你的职责是根据用户意图决策是否调用子工具并输出最终结果。

    工具策略：
    1. 订单相关问题调用 order_tool_agent。
    2. 商品相关问题调用 product_tool_agent。
    3. 可同时调用多个工具并做统一总结。
    4. 非业务闲聊可直接回答，不必强制调用工具。

    强约束：
    1. 严禁编造工具未返回的数据。
    2. 优先调用工具拿真实数据，再生成最终答复。
    3. 输出简洁清晰，不暴露内部调度细节。
""" + base_prompt


@traceable(name="Supervisor Agent Node", run_type="chain")
def supervisor_agent(state: AgentState) -> dict[str, Any]:
    llm = create_chat_model(
        model="qwen-flash",
        temperature=1.3,
    )
    history_messages = list(state.get("history_messages") or [])

    messages = [SystemMessage(content=_SUPERVISOR_PROMPT), *history_messages]
    trace = invoke_with_trace(
        llm,
        messages,
        tools=[order_tool_agent, product_tool_agent],
    )
    text = str(trace.get("text") or "").strip()
    trace_item = ExecutionTraceState(
        node_name="supervisor_agent",
        model_name=str(trace.get("model_name") or "unknown"),
        output_text=text,
        llm_used=True,
        llm_usage_complete=bool(trace.get("is_usage_complete", False)),
        llm_token_usage=trace.get("usage"),
        tool_calls=list(trace.get("tool_calls") or []),
    )
    execution_traces, token_usage = append_trace_and_refresh_token_usage(
        state.get("execution_traces"),
        trace_item,
    )
    return {
        "result": text,
        "messages": [AIMessage(content=text)],
        "execution_traces": execution_traces,
        "token_usage": token_usage,
    }
