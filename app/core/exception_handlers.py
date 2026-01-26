from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.codes import ResponseCode
from app.core.exceptions import ServiceException
from app.schemas.response import ApiResponse


class ExceptionHandlers:
    @staticmethod
    async def service_exception_handler(_: Request, exc: ServiceException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.code,
            content=ApiResponse.error(message=exc.message, code=exc.code, data=exc.data).model_dump(),
        )

    @staticmethod
    async def http_exception_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ApiResponse.error(message=str(exc.detail), code=exc.status_code).model_dump(),
        )

    @staticmethod
    async def unhandled_exception_handler(_: Request, __: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=ResponseCode.INTERNAL_ERROR,
            content=ApiResponse.error(
                message=ResponseCode.INTERNAL_ERROR.message,
                code=ResponseCode.INTERNAL_ERROR,
            ).model_dump(),
        )
