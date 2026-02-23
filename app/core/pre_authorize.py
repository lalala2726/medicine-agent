from __future__ import annotations

import inspect
from functools import wraps
from typing import Callable

from app.core.codes import ResponseCode
from app.exception.exceptions import ServiceException
from app.core.request_context import get_current_user
from app.core.role_codes import RoleCode

FORBIDDEN_MESSAGE = "无权限访问此接口"
INVALID_PREDICATE_MESSAGE = (
    "pre_authorize 需要可调用对象，请使用 lambda，例如："
    "pre_authorize(lambda: has_role(RoleCode.ADMIN) or "
    'has_permission("admin:assistant:access"))'
)


def has_role(role_code: str | RoleCode) -> bool:
    """校验当前用户是否拥有指定角色（严格精确匹配）。"""
    resolved_role = role_code.value if isinstance(role_code, RoleCode) else role_code
    return resolved_role in get_current_user().roles


def has_permission(permission_code: str) -> bool:
    """校验当前用户是否拥有指定权限（严格精确匹配）。"""
    return permission_code in get_current_user().permissions


def _check_predicate(predicate: Callable[[], bool]) -> None:
    result = predicate()
    if inspect.isawaitable(result):
        raise TypeError("pre_authorize 的 predicate 必须同步返回 bool")
    if not isinstance(result, bool):
        raise TypeError(
            "pre_authorize 的 predicate 必须返回 bool，"
            "请使用 lambda 包裹组合表达式。",
        )
    if not result:
        raise ServiceException(
            code=ResponseCode.FORBIDDEN,
            message=FORBIDDEN_MESSAGE,
        )


def pre_authorize(predicate: Callable[[], bool]):
    """
    基于上下文用户角色/权限的接口访问控制装饰器。

    示例：
        @pre_authorize(lambda: has_role(RoleCode.ADMIN) or has_permission("admin:assistant:access"))
    """
    if not callable(predicate):
        raise TypeError(INVALID_PREDICATE_MESSAGE)

    def _decorate(func):
        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def _async_wrapper(*args, **kwargs):
                _check_predicate(predicate)
                return await func(*args, **kwargs)

            return _async_wrapper

        @wraps(func)
        def _wrapper(*args, **kwargs):
            _check_predicate(predicate)
            return func(*args, **kwargs)

        return _wrapper

    return _decorate


__all__ = ["pre_authorize", "has_role", "has_permission", "RoleCode"]
