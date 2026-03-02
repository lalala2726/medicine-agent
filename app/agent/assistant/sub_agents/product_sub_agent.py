from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from app.agent.assistant.tools.product_tools import (
    get_drug_detail,
    get_product_detail,
    get_product_list,
)
from app.core.agent.agent_runtime import agent_invoke
from app.core.agent.agent_tool_events import tool_call_status
from app.core.langsmith import traceable
from app.core.llms import create_chat_model
from app.utils.prompt_utils import load_prompt

_PRODUCT_SYSTEM_PROMPT = load_prompt("assistant/sub_agents/product_sub_agent_system_prompt.md")


@tool(
    description=(
            "处理商品相关任务：商品列表、商品详情。"
            "输入为自然语言任务描述，内部会自动调用商品工具并返回结果。"
    )
)
@tool_call_status(
    tool_name="正在调用商品子代理",
    start_message="正在执行查询",
    error_message="调用商品子代理失败",
    timely_message="商品子代理正在持续处理中",
)
@traceable(name="Supervisor Product Sub-Agent", run_type="chain")
def product_sub_agent(task_description: str) -> str:
    """
    功能描述：
        执行商品域子代理，按任务描述调度商品工具并返回查询结果。

    参数说明：
        task_description (str): 子代理任务描述文本，通常由 Supervisor 节点生成。

    返回值：
        str: 子代理执行结果文本；若结果为空则返回默认提示“暂无数据”。

    异常说明：
        不主动抛出业务异常；底层工具或模型异常由上层统一处理与记录。
    """

    llm = create_chat_model(
        temperature=1.0,
    )
    agent = create_agent(
        model=llm,
        system_prompt=SystemMessage(content=_PRODUCT_SYSTEM_PROMPT),
        tools=[get_product_list, get_product_detail, get_drug_detail],
    )
    input_messages = str(task_description or "").strip()
    result = agent_invoke(
        agent,
        input_messages,
    )
    content = str(result.content or "").strip()
    return content or "暂无数据"
