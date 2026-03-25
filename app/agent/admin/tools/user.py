"""
用户领域工具。
"""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.agent.admin.tools.cache import save_current_admin_tool_cache_entry
from app.core.agent.agent_tool_events import tool_call_status
from app.schemas.http_response import HttpResponse
from app.utils.http_client import HttpClient


class UserListRequest(BaseModel):
    """
    功能描述：
        用户列表查询入参模型。

    参数说明：
        page_num (int): 页码。
        page_size (int): 每页数量。
        id (int | None): 用户 ID。
        username (str | None): 用户名。
        nickname (str | None): 昵称。
        avatar (str | None): 头像 URL。
        roles (str | None): 角色编码。
        status (int | None): 用户状态。
        create_by (str | None): 创建人。

    返回值：
        无（数据模型定义）。

    异常说明：
        无。
    """

    page_num: int = Field(default=1, ge=1, description="页码，从 1 开始")
    page_size: int = Field(default=10, ge=1, le=200, description="每页数量，范围 1-200")
    id: Optional[int] = Field(default=None, ge=1, description="用户 ID，精确匹配")
    username: Optional[str] = Field(default=None, description="用户名，支持模糊查询")
    nickname: Optional[str] = Field(default=None, description="昵称，支持模糊查询")
    avatar: Optional[str] = Field(default=None, description="头像 URL，精确匹配")
    roles: Optional[str] = Field(default=None, description="角色编码，例如 admin")
    status: Optional[int] = Field(default=None, description="用户状态，例如 1 启用、0 禁用")
    create_by: Optional[str] = Field(default=None, description="创建人")


class UserIdRequest(BaseModel):
    """
    功能描述：
        单个用户查询入参模型。

    参数说明：
        user_id (int): 用户 ID。

    返回值：
        无（数据模型定义）。

    异常说明：
        无。
    """

    user_id: int = Field(ge=1, description="用户 ID")


class UserIdPageRequest(UserIdRequest):
    """
    功能描述：
        带分页的用户查询入参模型。

    参数说明：
        user_id (int): 用户 ID。
        page_num (int): 页码。
        page_size (int): 每页数量。

    返回值：
        无（数据模型定义）。

    异常说明：
        无。
    """

    page_num: int = Field(default=1, ge=1, description="页码，从 1 开始")
    page_size: int = Field(default=10, ge=1, le=200, description="每页数量，范围 1-200")


@tool(
    args_schema=UserListRequest,
    description=(
            "分页查询用户列表，支持按用户 ID、用户名、昵称、角色、状态和创建人筛选。"
            "适用于定位目标用户范围。"
    ),
)
@tool_call_status(
    tool_name="查询用户列表",
    start_message="正在查询用户列表",
    error_message="查询用户列表失败",
    timely_message="用户列表正在持续处理中",
)
async def user_list(
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
    """
    功能描述：
        查询用户列表。

    参数说明：
        page_num (int): 页码。
        page_size (int): 每页数量。
        id (Optional[int]): 用户 ID。
        username (Optional[str]): 用户名。
        nickname (Optional[str]): 昵称。
        avatar (Optional[str]): 头像 URL。
        roles (Optional[str]): 角色编码。
        status (Optional[int]): 用户状态。
        create_by (Optional[str]): 创建人。

    返回值：
        dict: 用户列表数据。

    异常说明：
        无。
    """

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
        result = HttpResponse.parse_data(response)
        save_current_admin_tool_cache_entry(
            tool_name="user_list",
            tool_input=params,
            tool_output=result,
        )
        return result


@tool(
    args_schema=UserIdRequest,
    description=(
            "根据用户 ID 查询用户详情。"
            "适用于查看用户资料、角色和账号基础信息。"
    ),
)
@tool_call_status(
    tool_name="查询用户详情",
    start_message="正在查询用户详情",
    error_message="查询用户详情失败",
    timely_message="用户详情正在持续处理中",
)
async def user_detail(user_id: int) -> dict:
    """
    功能描述：
        根据用户 ID 查询用户详情。

    参数说明：
        user_id (int): 用户 ID。

    返回值：
        dict: 用户详情数据。

    异常说明：
        无。
    """

    async with HttpClient() as client:
        response = await client.get(url=f"/agent/admin/user/{user_id}/detail")
        result = HttpResponse.parse_data(response)
        save_current_admin_tool_cache_entry(
            tool_name="user_detail",
            tool_input={"user_id": user_id},
            tool_output=result,
        )
        return result


@tool(
    args_schema=UserIdRequest,
    description=(
            "根据用户 ID 查询钱包信息。"
            "适用于查看用户钱包余额和可用状态。"
    ),
)
@tool_call_status(
    tool_name="查询用户钱包",
    start_message="正在查询用户钱包",
    error_message="查询用户钱包失败",
    timely_message="用户钱包正在持续处理中",
)
async def user_wallet(user_id: int) -> dict:
    """
    功能描述：
        根据用户 ID 查询钱包信息。

    参数说明：
        user_id (int): 用户 ID。

    返回值：
        dict: 用户钱包数据。

    异常说明：
        无。
    """

    async with HttpClient() as client:
        response = await client.get(url=f"/agent/admin/user/{user_id}/wallet")
        result = HttpResponse.parse_data(response)
        save_current_admin_tool_cache_entry(
            tool_name="user_wallet",
            tool_input={"user_id": user_id},
            tool_output=result,
        )
        return result


@tool(
    args_schema=UserIdPageRequest,
    description=(
            "根据用户 ID 分页查询钱包流水。"
            "适用于查看用户钱包变动明细。"
    ),
)
@tool_call_status(
    tool_name="查询用户钱包流水",
    start_message="正在查询用户钱包流水",
    error_message="查询用户钱包流水失败",
    timely_message="用户钱包流水正在持续处理中",
)
async def user_wallet_flow(
        user_id: int,
        page_num: int = 1,
        page_size: int = 10,
) -> dict:
    """
    功能描述：
        根据用户 ID 分页查询钱包流水。

    参数说明：
        user_id (int): 用户 ID。
        page_num (int): 页码。
        page_size (int): 每页数量。

    返回值：
        dict: 用户钱包流水数据。

    异常说明：
        无。
    """

    async with HttpClient() as client:
        params = {
            "pageNum": page_num,
            "pageSize": page_size,
        }
        response = await client.get(
            url=f"/agent/admin/user/{user_id}/wallet_flow",
            params=params,
        )
        result = HttpResponse.parse_data(response)
        save_current_admin_tool_cache_entry(
            tool_name="user_wallet_flow",
            tool_input={"user_id": user_id, "page_num": page_num, "page_size": page_size},
            tool_output=result,
        )
        return result


@tool(
    args_schema=UserIdPageRequest,
    description=(
            "根据用户 ID 分页查询消费信息。"
            "适用于查看用户消费记录和消费明细。"
    ),
)
@tool_call_status(
    tool_name="查询用户消费信息",
    start_message="正在查询用户消费信息",
    error_message="查询用户消费信息失败",
    timely_message="用户消费信息正在持续处理中",
)
async def user_consume_info(
        user_id: int,
        page_num: int = 1,
        page_size: int = 10,
) -> dict:
    """
    功能描述：
        根据用户 ID 分页查询消费信息。

    参数说明：
        user_id (int): 用户 ID。
        page_num (int): 页码。
        page_size (int): 每页数量。

    返回值：
        dict: 用户消费信息数据。

    异常说明：
        无。
    """

    async with HttpClient() as client:
        params = {
            "pageNum": page_num,
            "pageSize": page_size,
        }
        response = await client.get(
            url=f"/agent/admin/user/{user_id}/consume_info",
            params=params,
        )
        result = HttpResponse.parse_data(response)
        save_current_admin_tool_cache_entry(
            tool_name="user_consume_info",
            tool_input={"user_id": user_id, "page_num": page_num, "page_size": page_size},
            tool_output=result,
        )
        return result


__all__ = [
    "UserIdPageRequest",
    "UserIdRequest",
    "UserListRequest",
    "user_consume_info",
    "user_detail",
    "user_list",
    "user_wallet",
    "user_wallet_flow",
]
