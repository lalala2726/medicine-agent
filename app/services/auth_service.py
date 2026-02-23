from __future__ import annotations

from typing import Any

import httpx
from pydantic import ValidationError

from app.core.codes import ResponseCode
from app.exception.exceptions import ServiceException
from app.schemas.auth import AuthUser, AuthorizationContext
from app.schemas.http_response import HttpResponse
from app.utils.http_client import HttpClient

AUTH_SERVICE_UNAVAILABLE_MESSAGE = "认证服务暂不可用，请稍后重试"


def _is_auth_failure_code(code: int) -> bool:
    """
    认证失败码判定：
    - 标准 HTTP 401/403
    - 业务扩展码 401x/403x（例如 4011: 访问令牌已过期，4012: 刷新令牌过期）
    """
    if code in {ResponseCode.UNAUTHORIZED.code, ResponseCode.FORBIDDEN.code}:
        return True
    code_text = str(code)
    return code_text.startswith("4011") or code_text.startswith("4012")


def _build_auth_user(payload: Any) -> AuthUser:
    try:
        context = AuthorizationContext.model_validate(payload)
    except ValidationError as exc:
        raise ServiceException(
            code=503,
            message="无法获取当前用户信息～请检查你是否登陆？",
        ) from exc

    return context.to_auth_user()


async def verify_authorization() -> AuthUser:
    """
    使用当前请求 Authorization 调用 Spring `/agent/authorization` 获取用户上下文。

    处理规则：
    - 通过 `/agent/authorization` 获取响应并解析 data。
    - `payload = HttpResponse.parse_data(response)` 的异常直接透传。
    - data 需满足 `{user, roles, permissions}` 结构（user 必填；roles/permissions 可空）。
    - 网络错误/超时仍映射为 HTTP 503。
    """
    try:
        async with HttpClient(timeout=5.0) as client:
            response = await client.get("/agent/authorization")
    except ServiceException as exc:
        if _is_auth_failure_code(exc.code):
            raise ServiceException(
                code=ResponseCode.UNAUTHORIZED,
                message=exc.message,
                data=exc.data,
            ) from exc
        raise ServiceException(
            code=503,
            message=AUTH_SERVICE_UNAVAILABLE_MESSAGE,
        ) from exc
    except (httpx.TimeoutException, httpx.HTTPError) as exc:
        raise ServiceException(
            code=503,
            message=AUTH_SERVICE_UNAVAILABLE_MESSAGE,
        ) from exc

    payload = HttpResponse.parse_data(response)

    return _build_auth_user(payload)
