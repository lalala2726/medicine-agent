from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from app.agent.assistant.tools.order_tools import (
    get_order_list,
    get_order_shipping,
    get_order_timeline,
    get_orders_detail,
)
from app.core.agent.agent_runtime import agent_invoke
from app.core.agent.agent_tool_events import tool_call_status
from app.core.langsmith import traceable
from app.core.llms import create_chat_model
from app.utils.prompt_utils import load_prompt

_ORDER_SYSTEM_PROMPT = load_prompt("assistant/sub_agents/order_sub_agent_system_prompt.md")


@tool(
    description=(
            "处理订单相关任务：订单列表、订单详情、订单流程、发货记录。"
            "输入为自然语言任务描述，内部会自动调用订单工具并返回结果。"
    )
)
@tool_call_status(
    tool_name="正在调用订单子代理",
    start_message="正在执行查询",
    error_message="调用订单子代理失败",
    timely_message="订单子代理正在持续处理中",
)
@traceable(name="Supervisor Order Sub-Agent", run_type="chain")
def order_sub_agent(task_description: str) -> str:
    """
    功能描述：
        执行订单域子代理，按任务描述调度订单工具并返回聚合后的文本结果。

    参数说明：
        task_description (str): 子代理任务描述文本，通常由 Supervisor 节点生成。

    返回值：
        str: 子代理执行结果文本；若结果为空则返回默认提示“暂无数据”。

    异常说明：
        不主动抛出业务异常；底层工具或模型异常由上层统一处理与记录。
    """

    llm = create_chat_model(
        model="qwen-flash",
        temperature=1.0,
    )
    agent = create_agent(
        model=llm,
        system_prompt=SystemMessage(content=_ORDER_SYSTEM_PROMPT),
        tools=[
            get_order_list,
            get_orders_detail,
            get_order_timeline,
            get_order_shipping,
        ],
    )
    input_messages = str(task_description or "").strip()
    result = agent_invoke(
        agent,
        input_messages,
    )
    content = str(result.content or "").strip()
    return content or "暂无数据"
