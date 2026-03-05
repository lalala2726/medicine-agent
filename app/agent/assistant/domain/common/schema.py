"""
基础工具数据模型。
"""

from __future__ import annotations

from pydantic import BaseModel

from app.schemas.auth import AuthUser


class UserInfo(BaseModel):
    """安全的用户信息模型，仅包含可暴露给 Agent 的非敏感字段。"""

    username: str | None = None
    nickname: str | None = None

    @classmethod
    def from_auth_user(cls, auth_user: AuthUser) -> "UserInfo":
        """从 AuthUser 创建 UserInfo，过滤敏感信息。"""
        return cls(
            username=auth_user.username,
            nickname=auth_user.nickname,
        )


__all__ = ["UserInfo"]
