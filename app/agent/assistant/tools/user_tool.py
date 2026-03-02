from __future__ import annotations

from typing import Optional

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from app.agent.assistant.tools.schemas.user import (
    AdminUserIdPageRequest,
    AdminUserIdRequest,
    AdminUserListQueryRequest,
)
from app.core.agent.agent_runtime import agent_invoke
from app.core.agent.agent_tool_events import tool_call_status
from app.core.langsmith import traceable
from app.core.llms import LlmProvider, create_chat_model
from app.schemas.http_response import HttpResponse
from app.utils.http_client import HttpClient
from app.utils.prompt_utils import load_prompt


@tool(
    args_schema=AdminUserListQueryRequest,
    description=(
            "分页查询管理端用户列表，支持按用户 ID、用户名、昵称、角色、状态、创建人等条件筛选。"
            "参数传递规则：使用结构化字段，不要把多个条件拼成单字符串。"
    ),
)
@tool_call_status(
    tool_name="查询用户列表",
    start_message="正在查询用户列表",
    error_message="查询用户列表失败",
    timely_message="用户列表正在持续处理中",
)
async def get_admin_user_list(
        page_num: int = 1,
        page_size: int = 10,
        id: Optional[int] = None,
        username: Optional[str] = None,
        nickname: Optional[str] = None,
        avatar: Optional[str] = None,
        roles: Optional[str] = None,
        status: Optional[int] = None,
        create_by: Optional[str] = None,
) -> dict:
    """分页查询管理端用户列表。"""

    async with HttpClient() as client:
        params = {
            "pageNum": page_num,
            "pageSize": page_size,
            "id": id,
            "username": username,
            "nickname": nickname,
            "avatar": avatar,
            "roles": roles,
            "status": status,
            "createBy": create_by,
        }
        response = await client.get(
            url="/agent/admin/user/list",
            params=params,
        )
        return HttpResponse.parse_data(response)


@tool(
    args_schema=AdminUserIdRequest,
    description=(
            "根据用户 ID 查询用户详情。"
            "调用时机：需要查看某个用户的完整资料、角色或账户基础信息时。"
    ),
)
@tool_call_status(
    tool_name="查询用户详情",
    start_message="正在查询用户详情",
    error_message="查询用户详情失败",
    timely_message="用户详情正在持续处理中",
)
async def get_admin_user_detail(user_id: int) -> dict:
    """根据用户 ID 查询用户详情。"""

    async with HttpClient() as client:
        response = await client.get(
            url=f"/agent/admin/user/{user_id}/detail",
        )
        return HttpResponse.parse_data(response)


@tool(
    args_schema=AdminUserIdRequest,
    description=(
            "根据用户 ID 查询钱包信息。"
            "返回钱包余额、可用状态等信息。"
    ),
)
@tool_call_status(
    tool_name="查询用户钱包",
    start_message="正在查询用户钱包",
    error_message="查询用户钱包失败",
    timely_message="用户钱包正在持续处理中",
)
async def get_admin_user_wallet(user_id: int) -> dict:
    """根据用户 ID 查询钱包信息。"""

    async with HttpClient() as client:
        response = await client.get(
            url=f"/agent/admin/user/{user_id}/wallet",
        )
        return HttpResponse.parse_data(response)


@tool(
    args_schema=AdminUserIdPageRequest,
    description=(
            "根据用户 ID 分页查询钱包流水。"
            "支持 page_num/page_size 分页参数，page_size 最大 200。"
    ),
)
@tool_call_status(
    tool_name="查询用户钱包流水",
    start_message="正在查询用户钱包流水",
    error_message="查询用户钱包流水失败",
    timely_message="用户钱包流水正在持续处理中",
)
async def get_admin_user_wallet_flow(
        user_id: int,
        page_num: int = 1,
        page_size: int = 10,
) -> dict:
    """根据用户 ID 分页查询钱包流水。"""

    async with HttpClient() as client:
        params = {
            "pageNum": page_num,
            "pageSize": page_size,
        }
        response = await client.get(
            url=f"/agent/admin/user/{user_id}/wallet_flow",
            params=params,
        )
        return HttpResponse.parse_data(response)


@tool(
    args_schema=AdminUserIdPageRequest,
    description=(
            "根据用户 ID 分页查询消费信息。"
            "支持 page_num/page_size 分页参数，page_size 最大 200。"
    ),
)
@tool_call_status(
    tool_name="查询用户消费信息",
    start_message="正在查询用户消费信息",
    error_message="查询用户消费信息失败",
    timely_message="用户消费信息正在持续处理中",
)
async def get_admin_user_consume_info(
        user_id: int,
        page_num: int = 1,
        page_size: int = 10,
) -> dict:
    """根据用户 ID 分页查询消费信息。"""

    async with HttpClient() as client:
        params = {
            "pageNum": page_num,
            "pageSize": page_size,
        }
        response = await client.get(
            url=f"/agent/admin/user/{user_id}/consume_info",
            params=params,
        )
        return HttpResponse.parse_data(response)


_USER_SYSTEM_PROMPT = load_prompt("assistant/user_system_prompt.md")


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
@traceable(name="Supervisor User Tool Agent", run_type="chain")
def user_tool_agent(task_description: str) -> str:
    llm = create_chat_model(
        model="qwen-flash",
        provider=LlmProvider.ALIYUN,
        temperature=0.2,
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
