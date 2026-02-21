from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, SystemMessage

from app.agent.admin.state import AgentState, ExecutionTraceState
from app.core.langsmith import traceable
from app.core.llm import create_chat_model
from app.schemas.prompt import base_prompt
from app.services.token_usage_service import append_trace_and_refresh_token_usage
from app.utils.streaming_utils import (
    invoke_with_trace,
    serialize_messages_for_trace,
)

_CHAT_SYSTEM_PROMPT = (
        """
            你是药品商城后台管理助手中的聊天节点（chat_agent）。
            你只处理闲聊、寒暄、通用说明，不负责订单/商品结果汇总。
            
            回复规则：
            1. 简洁、礼貌、自然，不要重复句子。
            2. 不要输出“我将调用工具”或内部调度细节。
            3. 若用户明显是业务查询（订单/商品/表格），给一句简短引导即可，不臆测数据。
    """
        + base_prompt
)


@traceable(name="Supervisor Chat Agent Node", run_type="chain")
def chat_agent(state: AgentState) -> dict[str, Any]:
    llm = create_chat_model(
        model="qwen-flash",
        temperature=1.3,
    )
    history_messages = list(state.get("history_messages") or [])

    messages = [SystemMessage(content=_CHAT_SYSTEM_PROMPT), *history_messages]
    trace = invoke_with_trace(llm, messages)
    text = str(trace.get("text") or "").strip()
    trace_item = ExecutionTraceState(
        node_name="chat_agent",
        model_name=str(trace.get("model_name") or "unknown"),
        input_messages=serialize_messages_for_trace(messages),
        output_text=text,
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
        "result": text,
        "messages": [AIMessage(content=text)],
        "execution_traces": execution_traces,
        "token_usage": token_usage,
    }
