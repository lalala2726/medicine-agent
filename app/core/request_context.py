from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Optional

from app.core.codes import ResponseCode
from app.core.exceptions import ServiceException
from app.schemas.auth import AuthUser

_authorization_header: ContextVar[Optional[str]] = ContextVar(
    "authorization_header",
    default=None,
)
_current_user: ContextVar[AuthUser | None] = ContextVar(
    "current_user",
    default=None,
)


def set_authorization_header(value: Optional[str]) -> Token:
    """在当前请求上下文中设置 Authorization 头。"""
    return _authorization_header.set(value)


def reset_authorization_header(token: Token) -> None:
    """重置 Authorization 头，避免跨请求污染。"""
    _authorization_header.reset(token)


def get_authorization_header() -> str:
    """获取当前请求的 Authorization 头。"""
    authorization = _authorization_header.get()
    if authorization is None:
        raise ServiceException(
            code=ResponseCode.UNAUTHORIZED,
            message="缺少 Authorization 请求头",
        )
    return authorization


def set_current_user(value: AuthUser | None) -> Token:
    """在当前请求上下文中设置用户信息。"""
    return _current_user.set(value)


def reset_current_user(token: Token) -> None:
    """重置用户上下文，避免跨请求污染。"""
    _current_user.reset(token)


def get_current_user() -> AuthUser:
    """获取当前请求上下文中的认证用户。"""
    user = _current_user.get()
    if user is None:
        raise ServiceException(
            code=ResponseCode.UNAUTHORIZED,
            message="当前请求未认证",
        )
    return user


def get_current_user_id() -> int:
    """获取当前请求上下文中的用户 ID。"""
    return get_current_user().id


def get_current_token() -> str:
    """获取当前请求的 Bearer token。"""
    authorization = get_authorization_header().strip()
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        raise ServiceException(
            code=ResponseCode.UNAUTHORIZED,
            message="无效 Authorization 请求头",
        )
    return parts[1].strip()


def get_user() -> AuthUser:
    """`get_current_user` 的简写别名。"""
    return get_current_user()


def get_user_id() -> int:
    """`get_current_user_id` 的简写别名。"""
    return get_current_user_id()
