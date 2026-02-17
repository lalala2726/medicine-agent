from __future__ import annotations

import httpx
from pydantic import ValidationError

from app.core.codes import ResponseCode
from app.core.exceptions import ServiceException
from app.schemas.auth import AuthUser
from app.schemas.http_response import HttpResponse
from app.utils.http_client import HttpClient

AUTH_SERVICE_UNAVAILABLE_MESSAGE = "认证服务暂不可用，请稍后重试"


def _is_auth_failure_code(code: int) -> bool:
    """
    认证失败码判定：
    - 标准 HTTP 401/403
    - 业务扩展码 401x/403x（例如 4011: 访问令牌已过期）
    """
    if code in {ResponseCode.UNAUTHORIZED.code, ResponseCode.FORBIDDEN.code}:
        return True
    code_text = str(code)
    return code_text.startswith("401") or code_text.startswith("403")


async def verify_authorization() -> AuthUser:
    """
    使用当前请求 Authorization 调用 Spring `/agent/authorization` 获取用户上下文。

    处理规则：
    - 上游返回 code=200：解析 data 为 AuthUser。
    - 上游返回 code!=200：
      - 401/403/401x/403x -> 映射为 HTTP 401（保留 message）。
      - 其余错误 -> 映射为 HTTP 503（保留 message）。
    - 网络错误/超时/响应非 JSON/data 结构不合法 -> HTTP 503。
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

    try:
        payload = HttpResponse.parse_data(response)
    except ServiceException as exc:
        if _is_auth_failure_code(exc.code):
            raise ServiceException(
                code=ResponseCode.UNAUTHORIZED,
                message=exc.message,
                data=exc.data,
            ) from exc
        if str(exc.message).startswith("响应不是合法 JSON"):
            raise ServiceException(code=503, message="认证服务响应格式错误") from exc
        raise ServiceException(
            code=503,
            message=exc.message or AUTH_SERVICE_UNAVAILABLE_MESSAGE,
            data=exc.data,
        ) from exc

    try:
        return AuthUser.model_validate(payload)
    except ValidationError as exc:
        raise ServiceException(
            code=503,
            message="认证服务响应用户信息格式错误",
        ) from exc
