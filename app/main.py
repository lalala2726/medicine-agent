from dotenv import load_dotenv
from fastapi import FastAPI
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.main import api_router
from app.core.exception_handlers import ExceptionHandlers
from app.core.exceptions import ServiceException

# 加载 .env 配置，确保本地开发环境变量生效
load_dotenv()

app = FastAPI()
app.include_router(api_router)

app.add_exception_handler(ServiceException, ExceptionHandlers.service_exception_handler)
app.add_exception_handler(StarletteHTTPException, ExceptionHandlers.http_exception_handler)
app.add_exception_handler(Exception, ExceptionHandlers.unhandled_exception_handler)
