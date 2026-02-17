from __future__ import annotations

import httpx
from pydantic import ValidationError

from app.core.codes import ResponseCode
from app.core.exceptions import ServiceException
from app.schemas.auth import AuthUser
from app.schemas.http_response import HttpResponse
from app.utils.http_client import HttpClient

AUTH_SERVICE_UNAVAILABLE_MESSAGE = "认证服务暂不可用，请稍后重试"


async def fetch_current_user_by_authorization() -> AuthUser:
    """
    通过当前请求头中的 Authorization 调用 Spring 鉴权接口获取用户。

    映射规则：
    - Spring 返回 code=401/403 -> FastAPI 抛 401
    - Spring 返回 code!=200 且非 401/403 -> FastAPI 抛 503
    - 网络错误/超时/非 JSON/无 data/data 不合法 -> FastAPI 抛 503
    """
    try:
        async with HttpClient(timeout=5.0) as client:
            response = await client.get("/agent/authorization")
    except ServiceException as exc:
        if exc.code == ResponseCode.UNAUTHORIZED.code:
            raise
        raise ServiceException(
            code=503,
            message=AUTH_SERVICE_UNAVAILABLE_MESSAGE,
        ) from exc
    except (httpx.TimeoutException, httpx.HTTPError) as exc:
        raise ServiceException(
            code=503,
            message=AUTH_SERVICE_UNAVAILABLE_MESSAGE,
        ) from exc

    try:
        payload = HttpResponse.from_response(response)
    except ServiceException as exc:
        raise ServiceException(
            code=503,
            message="认证服务响应格式错误",
        ) from exc

    if payload.code in {ResponseCode.UNAUTHORIZED.code, ResponseCode.FORBIDDEN.code}:
        raise ServiceException(
            code=ResponseCode.UNAUTHORIZED,
            message=payload.message or ResponseCode.UNAUTHORIZED.message,
        )

    if payload.code != ResponseCode.SUCCESS.code:
        raise ServiceException(
            code=503,
            message=AUTH_SERVICE_UNAVAILABLE_MESSAGE,
        )

    if payload.data is None:
        raise ServiceException(
            code=503,
            message="认证服务响应缺少用户信息",
        )

    try:
        return AuthUser.model_validate(payload.data)
    except ValidationError as exc:
        raise ServiceException(
            code=503,
            message="认证服务响应用户信息格式错误",
        ) from exc
