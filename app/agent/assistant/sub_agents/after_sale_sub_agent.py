from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from app.agent.assistant.tools.after_sale_tools import (
    get_admin_after_sale_detail,
    get_admin_after_sale_list,
)
from app.core.agent.agent_runtime import agent_invoke
from app.core.agent.agent_tool_events import tool_call_status
from app.core.langsmith import traceable
from app.core.llms import create_chat_model
from app.utils.prompt_utils import load_prompt

_AFTER_SALE_SYSTEM_PROMPT = load_prompt("assistant/after_sale_system_prompt.md")


@tool(
    description=(
            "处理管理端售后相关任务：售后列表、售后详情。"
            "输入为自然语言任务描述，内部会自动调用售后域工具并返回结果。"
    )
)
@tool_call_status(
    tool_name="正在调用售后子代理",
    start_message="正在执行售后查询",
    error_message="调用售后子代理失败",
    timely_message="售后子代理正在持续处理中",
)
@traceable(name="Supervisor After Sale Sub-Agent", run_type="chain")
def after_sale_sub_agent(task_description: str) -> str:
    """
    功能描述：
        执行售后域子代理，按任务描述调度售后工具并返回查询结果。

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
        system_prompt=SystemMessage(content=_AFTER_SALE_SYSTEM_PROMPT),
        tools=[
            get_admin_after_sale_list,
            get_admin_after_sale_detail,
        ],
    )
    input_messages = str(task_description or "").strip()
    result = agent_invoke(
        agent,
        input_messages,
    )
    content = str(result.content or "").strip()
    return content or "暂无数据"
