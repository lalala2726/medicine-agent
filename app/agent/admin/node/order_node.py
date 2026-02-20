from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.admin.agent_utils import execute_tool_node
from app.agent.admin.node.common import build_worker_update, serialize_messages
from app.agent.admin.state import AgentState
from app.agent.admin.tools.admin_tools import get_order_list, get_orders_detail
from app.core.assistant_status import status_node
from app.core.langsmith import traceable
from app.core.llm import create_chat_model
from app.schemas.prompt import base_prompt

_ORDER_SYSTEM_PROMPT = (
        """
你是药品商城后台的订单节点（order_agent）。
只处理订单相关任务，不要处理商品/闲聊。

执行要求：
1. 优先使用输入 context 的已提取信息（例如订单ID），避免重复追问用户。
2. 仅使用真实工具返回，不得编造。
3. 如果无结果，明确返回“未查到相关订单信息”。
4. 每次执行尽量给出可继续推进的结构化信息（如订单ID、商品ID）。

输入 instruction 是 JSON。
"""
        + base_prompt
)


def _build_instruction(state: AgentState) -> str:
    payload = {
        "user_input": state.get("user_input"),
        "context": state.get("context") or {},
    }
    return json.dumps(payload, ensure_ascii=False, default=str)


@status_node(
    node="order",
    start_message="正在处理订单问题",
    display_when="always",
)
@traceable(name="Supervisor Order Agent Node", run_type="chain")
def order_agent(state: AgentState) -> dict[str, Any]:
    """
    Execute order node in gateway + supervisor workflow.
    """
    instruction = _build_instruction(state)
    input_messages: list[Any] = [
        SystemMessage(content=_ORDER_SYSTEM_PROMPT),
        HumanMessage(content=instruction),
    ]
    llm = create_chat_model(model="qwen3-max")
    execution_result = execute_tool_node(
        llm=llm,
        messages=input_messages,
        tools=[get_order_list, get_orders_detail],
        enable_stream=True,
        failure_policy={},
        fallback_content="订单服务暂时不可用，请稍后重试。",
        fallback_error="order_agent 执行失败",
    )
    return build_worker_update(
        state=state,
        node_name="order_agent",
        result_key="order",
        content=execution_result.content,
        status=execution_result.status,
        model_name=execution_result.model_name,
        input_messages=serialize_messages(input_messages),
        tool_calls=execution_result.tool_calls,
        error=execution_result.error,
    )

