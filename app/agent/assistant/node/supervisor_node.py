from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, SystemMessage

from agent.assistant.model_switch import model_switch
from app.agent.assistant.state import AgentState, ExecutionTraceState
from app.agent.assistant.tools.analytics_tool import analytics_tool_agent
from app.agent.assistant.tools.chart_tool import chart_tool_agent
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
    3. 运营分析相关问题调用 analytics_tool_agent。
    4. 图表类型与模板相关问题调用 chart_tool_agent。
    5. 可同时调用多个工具并做统一总结。
    6. 非业务闲聊可直接回答，不必强制调用工具。
    
    ## 图表生成格式
    这边输出必须严格按照markdown代码块的方式进行输出否则前端无法进行渲染，比如你需要渲染**fishbonediagram**，这边前面必须是**```fishbonediagram**
    图表里面的value等参数必须严格按照工具返回结果进行填写。禁止出现任何违规的符号如数学运算符号等
    强约束：
    1. 严禁编造工具未返回的数据。
    2. 优先调用工具拿真实数据，再生成最终答复。
    3. 输出简洁清晰，不暴露内部调度细节。
""" + base_prompt


@traceable(name="Supervisor Agent Node", run_type="chain")
def supervisor_agent(state: AgentState) -> dict[str, Any]:
    model_name = model_switch(state)

    llm = create_chat_model(
        model=model_name,
        temperature=1.3,
    )
    history_messages = list(state.get("history_messages") or [])

    messages = [SystemMessage(content=_SUPERVISOR_PROMPT), *history_messages]
    trace = invoke_with_trace(
        llm,
        messages,
        tools=[order_tool_agent, product_tool_agent, analytics_tool_agent, chart_tool_agent],
    )
    text = str(trace.get("text") or "").strip()
    trace_item = ExecutionTraceState(
        node_name="supervisor_agent",
        model_name=model_name,
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
