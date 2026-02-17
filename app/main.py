from typing import Awaitable, Callable

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response

from app.api.main import api_router
from app.core.cors import load_cors_config
from app.core.exception_handlers import ExceptionHandlers
from app.core.exceptions import ServiceException
from app.core.request_context import (
    reset_authorization_header,
    reset_current_user,
    set_authorization_header,
    set_current_user,
)
from app.services.auth_service import fetch_current_user_by_authorization

# 加载 .env 配置，确保本地开发环境变量生效
load_dotenv()

app = FastAPI()
app.add_middleware(CORSMiddleware, **load_cors_config())
app.include_router(api_router)

app.add_exception_handler(
    RequestValidationError,
    ExceptionHandlers.request_validation_exception_handler,
)
app.add_exception_handler(ServiceException, ExceptionHandlers.service_exception_handler)
app.add_exception_handler(StarletteHTTPException, ExceptionHandlers.http_exception_handler)
app.add_exception_handler(Exception, ExceptionHandlers.unhandled_exception_handler)

AUTH_BYPASS_PATHS = {
    "/docs",
    "/redoc",
    "/openapi.json",
    "/docs/oauth2-redirect",
    "/favicon.ico",
}


def _should_skip_authorization(request: Request) -> bool:
    if request.method.upper() == "OPTIONS":
        return True
    path = request.url.path
    if path in AUTH_BYPASS_PATHS:
        return True
    if path.startswith("/docs/") or path.startswith("/redoc/"):
        return True
    return False


@app.middleware("http")
async def authorization_header_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    auth_token = set_authorization_header(request.headers.get("Authorization"))
    user_token = set_current_user(None)
    try:
        if not _should_skip_authorization(request):
            try:
                current_user = await fetch_current_user_by_authorization()
            except ServiceException as exc:
                return await ExceptionHandlers.service_exception_handler(request, exc)
            except Exception as exc:
                return await ExceptionHandlers.unhandled_exception_handler(request, exc)
            reset_current_user(user_token)
            user_token = set_current_user(current_user)
        return await call_next(request)
    finally:
        reset_current_user(user_token)
        reset_authorization_header(auth_token)
