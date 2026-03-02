from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from app.agent.assistant.tools.analytics_tools import (
    get_analytics_hot_products,
    get_analytics_order_status_distribution,
    get_analytics_order_trend,
    get_analytics_overview,
    get_analytics_payment_distribution,
    get_analytics_product_return_rates,
)
from app.core.agent.agent_runtime import agent_invoke
from app.core.agent.agent_tool_events import tool_call_status
from app.core.langsmith import traceable
from app.core.llms import create_chat_model
from app.utils.prompt_utils import load_prompt

_ANALYTICS_SYSTEM_PROMPT = load_prompt("assistant/analytics_system_prompt.md")


@tool(
    description=(
            "处理运营分析相关任务：总览、趋势、分布、排行榜。"
            "输入为自然语言任务描述，内部会自动调用分析工具并返回结果。"
    )
)
@tool_call_status(
    tool_name="正在调用运营分析子代理",
    start_message="正在执行查询",
    error_message="调用运营分析子代理失败",
    timely_message="运营分析子代理正在持续处理中",
)
@traceable(name="Supervisor Analytics Sub-Agent", run_type="chain")
def analytics_sub_agent(task_description: str) -> str:
    """
    功能描述：
        执行运营分析子代理，按任务描述调度运营分析工具并返回聚合结果。

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
        system_prompt=SystemMessage(content=_ANALYTICS_SYSTEM_PROMPT),
        tools=[
            get_analytics_overview,
            get_analytics_order_trend,
            get_analytics_order_status_distribution,
            get_analytics_payment_distribution,
            get_analytics_hot_products,
            get_analytics_product_return_rates,
        ],
    )
    input_messages = str(task_description or "").strip()
    result = agent_invoke(
        agent,
        input_messages,
    )
    content = str(result.content or "").strip()
    return content or "暂无数据"
