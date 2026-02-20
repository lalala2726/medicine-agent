from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.admin.agent_utils import execute_tool_node
from app.agent.admin.node.common import build_worker_update, serialize_messages
from app.agent.admin.state import AgentState
from app.agent.admin.tools.admin_tools import (
    get_drug_detail,
    get_product_detail,
    get_product_list,
)
from app.core.assistant_status import status_node
from app.core.langsmith import traceable
from app.core.llm import create_chat_model
from app.schemas.prompt import base_prompt

_PRODUCT_SYSTEM_PROMPT = (
        """
你是药品商城后台的商品节点（product_agent）。
只处理商品相关任务，不要处理订单/闲聊。

执行要求：
1. 优先使用输入 context 的已提取信息（例如 product_id）。
2. 仅使用真实工具返回，不得编造。
3. 如果无结果，明确返回“未查到相关商品信息”。
4. 尽量批量查询，不要对同类ID逐个重复调用。

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
    node="product",
    start_message="正在处理商品问题",
    display_when="always",
)
@traceable(name="Supervisor Product Agent Node", run_type="chain")
def product_agent(state: AgentState) -> dict[str, Any]:
    """
    Execute product node in gateway + supervisor workflow.
    """
    instruction = _build_instruction(state)
    input_messages: list[Any] = [
        SystemMessage(content=_PRODUCT_SYSTEM_PROMPT),
        HumanMessage(content=instruction),
    ]
    llm = create_chat_model(model="qwen3-max")
    execution_result = execute_tool_node(
        llm=llm,
        messages=input_messages,
        tools=[get_product_list, get_product_detail, get_drug_detail],
        enable_stream=True,
        failure_policy={},
        fallback_content="商品服务暂时不可用，请稍后重试。",
        fallback_error="product_agent 执行失败",
    )
    return build_worker_update(
        state=state,
        node_name="product_agent",
        result_key="product",
        content=execution_result.content,
        status=execution_result.status,
        model_name=execution_result.model_name,
        input_messages=serialize_messages(input_messages),
        tool_calls=execution_result.tool_calls,
        error=execution_result.error,
    )

