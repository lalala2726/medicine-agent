from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from app.agent.assistant.tools.user_tools import (
    get_admin_user_consume_info,
    get_admin_user_detail,
    get_admin_user_list,
    get_admin_user_wallet,
    get_admin_user_wallet_flow,
)
from app.core.agent.agent_runtime import agent_invoke
from app.core.agent.agent_tool_events import tool_call_status
from app.core.langsmith import traceable
from app.core.llms import create_chat_model
from app.utils.prompt_utils import load_prompt

_USER_SYSTEM_PROMPT = load_prompt("assistant/sub_agents/user_sub_agent_system_prompt.md")


@tool(
    description=(
            "处理管理端用户相关任务：用户列表、用户详情、用户钱包、钱包流水、消费信息。"
            "输入为自然语言任务描述，内部会自动调用用户域工具并返回结果。"
    )
)
@tool_call_status(
    tool_name="正在调用用户子代理",
    start_message="正在执行用户查询",
    error_message="调用用户子代理失败",
    timely_message="用户子代理正在持续处理中",
)
@traceable(name="Supervisor User Sub-Agent", run_type="chain")
def user_sub_agent(task_description: str) -> str:
    """
    功能描述：
        执行用户域子代理，按任务描述调度用户工具并返回查询结果。

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
        system_prompt=SystemMessage(content=_USER_SYSTEM_PROMPT),
        tools=[
            get_admin_user_list,
            get_admin_user_detail,
            get_admin_user_wallet,
            get_admin_user_wallet_flow,
            get_admin_user_consume_info,
        ],
    )
    input_messages = str(task_description or "").strip()
    result = agent_invoke(
        agent,
        input_messages,
    )
    content = str(result.content or "").strip()
    return content or "暂无数据"
