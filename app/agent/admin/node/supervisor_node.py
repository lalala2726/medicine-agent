from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, SystemMessage

from app.agent.admin.node.order_node import order_agent
from app.agent.admin.node.product_node import product_agent
from app.agent.admin.state import AgentState
from app.core.langsmith import traceable
from app.core.llm import create_chat_model
from app.schemas.prompt import base_prompt
from app.utils.streaming_utils import invoke

_SUPERVISOR_PROMPT = """
    你是药品商城后台管理助手的 supervisor 节点。
    你的职责是根据用户意图决策是否调用子工具并输出最终结果。

    工具策略：
    1. 订单相关问题调用 order_agent。
    2. 商品相关问题调用 product_agent。
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
        temperature=1.0,
    )
    history_messages = list(state.get("history_messages") or [])
    if not history_messages:
        text = "我在，请告诉我你想查询订单还是商品信息。"
        return {
            "result": text,
            "messages": [AIMessage(content=text)],
        }

    messages = [SystemMessage(content=_SUPERVISOR_PROMPT), *history_messages]

    content = invoke(
        llm,
        messages,
        tools=[order_agent, product_agent],
    )
    text = str(content or "").strip()
    return {
        "result": text,
        "messages": [AIMessage(content=text)],
    }
