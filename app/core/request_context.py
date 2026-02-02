from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Optional

from core.exceptions import ServiceException

_authorization_header: ContextVar[Optional[str]] = ContextVar(
    "authorization_header",
    default=None,
)


def set_authorization_header(value: Optional[str]) -> Token:
    """在当前请求上下文中设置 Authorization 头。"""
    return _authorization_header.set(value)


def reset_authorization_header(token: Token) -> None:
    """重置 Authorization 头，避免跨请求污染。"""
    _authorization_header.reset(token)


def get_authorization_header() -> Optional[str]:
    """获取当前请求的 Authorization 头。"""
    if _authorization_header.get() is None:
        raise ServiceException("Authorization header is missing")
    return _authorization_header.get()
