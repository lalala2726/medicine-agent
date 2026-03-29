import asyncio
from contextlib import asynccontextmanager
from typing import Awaitable, Callable

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pymongo.errors import PyMongoError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response

from app.api.main import api_router
from app.core.config_sync import initialize_agent_config_snapshot
from app.core.database import clear_neo4j_connection_cache, verify_neo4j_connection
from app.core.database.neo4j.config import is_neo4j_startup_ping_enabled
from app.core.exception.exception_handlers import ExceptionHandlers
from app.core.exception.exceptions import ServiceException
from app.core.mq.broker import get_broker, is_mq_configured
from app.core.security.anonymous_access import is_anonymous_request
from app.core.security.auth_context import (
    reset_authorization_header,
    reset_current_user,
    set_authorization_header,
    set_current_user,
)
from app.core.security.cors import load_cors_config
from app.core.security.system_auth import is_system_request, verify_system_request
from app.core.speech import (
    verify_volcengine_stt_connection_on_startup,
    verify_volcengine_tts_connection_on_startup,
)
from app.services.auth_service import verify_authorization

# 加载 .env 配置，确保本地开发环境变量生效
load_dotenv()

OPENAPI_DESCRIPTION = """
    ## 项目简介
    
    本项目提供药品相关的 AI 能力接口，包含管理助手对话、药品图片解析、知识库导入与检索等功能。
    
    ## 认证说明
    
    除 `/docs`、`/redoc`、`/openapi.json`、显式标注 `allow_anonymous`（匿名）与
    `allow_system`（系统签名）的接口外，其他接口均需要用户认证。
    
    请由药品服务端提供访问令牌，并在请求头中携带：
    
    - `Authorization: Bearer <token>`
"""

_speech_startup_probe_done = False


async def _run_speech_startup_probes() -> None:
    """在配置快照就绪后执行语音启动探活。"""

    probe_results = await asyncio.gather(
        verify_volcengine_stt_connection_on_startup(),
        verify_volcengine_tts_connection_on_startup(),
        return_exceptions=True,
    )
    for result in probe_results:
        if isinstance(result, Exception):
            raise result


async def _prepare_runtime_before_serving() -> None:
    """显式固化启动顺序：先加载配置快照，再探活语音连接。"""

    initialize_agent_config_snapshot()
    if is_neo4j_startup_ping_enabled():
        verify_neo4j_connection()
    await _run_speech_startup_probes()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _speech_startup_probe_done
    if not _speech_startup_probe_done:
        await _prepare_runtime_before_serving()
        _speech_startup_probe_done = True
    else:
        initialize_agent_config_snapshot()

    # 启动 MQ broker（有配置时才启动）
    _broker = None
    if is_mq_configured():
        import app.core.mq.handlers  # noqa: F401 — 触发 subscriber 注册
        _broker = get_broker()
        await _broker.start()

    try:
        yield
    finally:
        try:
            if _broker is not None:
                await _broker.close()
        finally:
            clear_neo4j_connection_cache()


app = FastAPI(
    title="Medicine AI Agent API",
    description=OPENAPI_DESCRIPTION,
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, **load_cors_config())
app.include_router(api_router)

app.add_exception_handler(
    RequestValidationError,
    ExceptionHandlers.request_validation_exception_handler,
)
app.add_exception_handler(ServiceException, ExceptionHandlers.service_exception_handler)
app.add_exception_handler(StarletteHTTPException, ExceptionHandlers.http_exception_handler)
app.add_exception_handler(PyMongoError, ExceptionHandlers.pymongo_exception_handler)
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
    if is_anonymous_request(request):
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
            if is_system_request(request):
                try:
                    await verify_system_request(request)
                except ServiceException as exc:
                    return await ExceptionHandlers.service_exception_handler(request, exc)
                except Exception as exc:
                    return await ExceptionHandlers.unhandled_exception_handler(request, exc)
            else:
                try:
                    current_user = await verify_authorization()
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
