from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.codes import ResponseCode
from app.core.exceptions import ServiceException
from app.schemas.response import ApiResponse


class ExceptionHandlers:
    @staticmethod
    async def service_exception_handler(_: Request, exc: ServiceException) -> JSONResponse:
        try:
            response_code = ResponseCode(exc.code)
        except ValueError:
            response_code = None
        if response_code:
            content = ApiResponse.error(response_code, message=exc.message, data=exc.data).model_dump()
        else:
            content = ApiResponse(code=exc.code, message=exc.message, data=exc.data).model_dump()
        return JSONResponse(
            status_code=exc.code,
            content=content,
        )

    @staticmethod
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        try:
            response_code = ResponseCode(exc.status_code)
        except ValueError:
            response_code = None
        if response_code:
            # 404 错误时包含请求路径信息
            if exc.status_code == 404:
                message = f"路由 {request.url.path} 不存在"
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
    async def unhandled_exception_handler(_: Request, __: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=ResponseCode.INTERNAL_ERROR,
            content=ApiResponse.error(ResponseCode.INTERNAL_ERROR).model_dump(),
        )
