from typing import Awaitable, Callable

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response

from app.api.main import api_router
from app.core.exception_handlers import ExceptionHandlers
from app.core.exceptions import ServiceException
from app.core.request_context import (
    reset_authorization_header,
    set_authorization_header,
)

# 加载 .env 配置，确保本地开发环境变量生效
load_dotenv()

app = FastAPI()
app.include_router(api_router)

app.add_exception_handler(ServiceException, ExceptionHandlers.service_exception_handler)
app.add_exception_handler(StarletteHTTPException, ExceptionHandlers.http_exception_handler)
app.add_exception_handler(Exception, ExceptionHandlers.unhandled_exception_handler)


@app.middleware("http")
async def authorization_header_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    token = set_authorization_header(request.headers.get("Authorization"))
    try:
        return await call_next(request)
    finally:
        reset_authorization_header(token)
