"""
用户工具参数模型。
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class AdminUserListQueryRequest(BaseModel):
    """管理端用户列表查询请求参数。"""

    page_num: int = Field(default=1, ge=1, description="页码，从 1 开始")
    page_size: int = Field(default=10, ge=1, le=200, description="每页数量，范围 1-200")
    id: Optional[int] = Field(default=None, ge=1, description="用户 ID，精确匹配")
    username: Optional[str] = Field(default=None, description="用户名，支持模糊查询")
    nickname: Optional[str] = Field(default=None, description="昵称，支持模糊查询")
    avatar: Optional[str] = Field(default=None, description="头像 URL，精确匹配")
    roles: Optional[str] = Field(default=None, description="角色编码，例如 admin")
    status: Optional[int] = Field(default=None, description="用户状态，例如 1 启用、0 禁用")
    create_by: Optional[str] = Field(default=None, description="创建人")


class AdminUserIdRequest(BaseModel):
    """按用户 ID 查询请求参数。"""

    user_id: int = Field(ge=1, description="用户 ID")


class AdminUserIdPageRequest(AdminUserIdRequest):
    """按用户 ID + 分页查询请求参数。"""

    page_num: int = Field(default=1, ge=1, description="页码，从 1 开始")
    page_size: int = Field(default=10, ge=1, le=200, description="每页数量，范围 1-200")


__all__ = [
    "AdminUserIdPageRequest",
    "AdminUserIdRequest",
    "AdminUserListQueryRequest",
]
