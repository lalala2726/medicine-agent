from __future__ import annotations

from typing import Any

from langchain_core.messages import SystemMessage

from app.agent.admin.state import AgentState
from app.core.langsmith import traceable
from core.llm import create_chat_model

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
    
    示例：
    - 用户: "在吗"
      输出: {"route_target":"chat_agent","task_difficulty":"simple"}
    - 用户: "把上个月退款超过2次的订单找出来"
      输出: {"route_target":"supervisor_agent","task_difficulty":"complex"}
    """


@traceable(name="Supervisor Gateway Router Node", run_type="chain")
def gateway_router(state: AgentState) -> AgentState:
    llm = create_chat_model(
        model="qwen-flash",
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    messages = [
        SystemMessage(content=_GATEWAY_PROMPT),
        state.get('history_messages')
    ]
    res = llm.invoke(messages)
    state['router'] = res.content
    print("当前的路由的目标是",state)
    return state
