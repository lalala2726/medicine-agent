from __future__ import annotations

from app.core.request_context import get_user, get_user_id
from app.schemas.auth import AuthUser


def get_current_user() -> AuthUser:
    """获取当前请求上下文中的用户。"""
    return get_user()


def get_current_user_id() -> int:
    """获取当前请求上下文中的用户 ID。"""
    return get_user_id()


def get_user_info() -> AuthUser:
    """兼容旧命名：返回当前用户信息。"""
    return get_current_user()
