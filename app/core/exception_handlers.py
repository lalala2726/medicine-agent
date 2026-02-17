import time

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.codes import ResponseCode
from app.core.exceptions import ServiceException
from app.schemas.response import ApiResponse


class ExceptionHandlers:
    @staticmethod
    def _build_request_context(request: Request | None) -> str:
        """
        构造统一的请求定位信息。

        仅输出 method/path/client，避免泄露 body 与敏感头信息。
        """

        if request is None:
            return "method=unknown path=unknown client=unknown"

        method = request.method or "unknown"
        path = request.url.path or "unknown"
        client = request.client
        if client is None:
            client_text = "unknown"
        elif client.port is None:
            client_text = client.host
        else:
            client_text = f"{client.host}:{client.port}"
        return f"method={method} path={path} client={client_text}"

    @staticmethod
    def _log_exception(*, request: Request | None, exc: Exception, category: str) -> None:
        """
        记录异常详细日志（含堆栈）用于排查。

        category 用于标识异常来源：validation/service/http/unhandled。
        """

        request_context = ExceptionHandlers._build_request_context(request)
        logger.opt(exception=exc).error(
            "API exception category={} {}",
            category,
            request_context,
        )

    @staticmethod
    def _resolve_http_status(code: int) -> int:
        """
        JSONResponse 仅接受合法 HTTP 状态码（100-599）。
        当业务 code 非法（如 4011）时，回落为 400，避免 ASGI 层抛错。
        """
        return code if 100 <= int(code) <= 599 else ResponseCode.BAD_REQUEST.code

    @staticmethod
    def _format_validation_errors(exc: RequestValidationError) -> list[dict[str, str]]:
        formatted_errors: list[dict[str, str]] = []
        for item in exc.errors():
            loc = item.get("loc", ())
            field_parts = [
                str(part)
                for part in loc
                if part not in {"body", "query", "path", "header", "cookie"}
            ]
            field = ".".join(field_parts) if field_parts else "body"
            formatted_errors.append(
                {
                    "field": field,
                    "message": item.get("msg", "Validation error"),
                    "type": item.get("type", "validation_error"),
                }
            )
        return formatted_errors

    @staticmethod
    async def request_validation_exception_handler(
            request: Request | None, exc: RequestValidationError
    ) -> JSONResponse:
        ExceptionHandlers._log_exception(
            request=request,
            exc=exc,
            category="validation",
        )
        return JSONResponse(
            status_code=ResponseCode.BAD_REQUEST,
            content={
                "code": ResponseCode.BAD_REQUEST,
                "message": "Validation Failed",
                "errors": ExceptionHandlers._format_validation_errors(exc),
                "timestamp": int(time.time() * 1000),
            },
        )

    @staticmethod
    async def service_exception_handler(request: Request | None, exc: ServiceException) -> JSONResponse:
        ExceptionHandlers._log_exception(
            request=request,
            exc=exc,
            category="service",
        )
        try:
            response_code = ResponseCode(exc.code)
        except ValueError:
            response_code = None
        if response_code:
            content = ApiResponse.error(response_code, message=exc.message, data=exc.data).model_dump()
        else:
            content = ApiResponse(code=exc.code, message=exc.message, data=exc.data).model_dump()
        return JSONResponse(
            status_code=ExceptionHandlers._resolve_http_status(exc.code),
            content=content,
        )

    @staticmethod
    async def http_exception_handler(request: Request | None, exc: StarletteHTTPException) -> JSONResponse:
        ExceptionHandlers._log_exception(
            request=request,
            exc=exc,
            category="http",
        )
        try:
            response_code = ResponseCode(exc.status_code)
        except ValueError:
            response_code = None
        if response_code:
            # 404 错误时包含请求路径信息
            if exc.status_code == 404:
                path = request.url.path if request is not None else "unknown"
                message = f"路由 {path} 不存在"
            else:
                message = None
            content = ApiResponse.error(response_code, message=message).model_dump()
        else:
            content = ApiResponse(code=exc.status_code, message=str(exc.detail)).model_dump()
        return JSONResponse(
            status_code=exc.status_code,
            content=content,
        )

    @staticmethod
    async def unhandled_exception_handler(request: Request | None, exc: Exception) -> JSONResponse:
        ExceptionHandlers._log_exception(
            request=request,
            exc=exc,
            category="unhandled",
        )
        return JSONResponse(
            status_code=ResponseCode.INTERNAL_ERROR,
            content=ApiResponse.error(ResponseCode.INTERNAL_ERROR).model_dump(),
        )
