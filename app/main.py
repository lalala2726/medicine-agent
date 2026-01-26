from fastapi import FastAPI
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.main import api_router
from app.core.exceptions import ServiceException
from app.core.exception_handlers import ExceptionHandlers

app = FastAPI()
app.include_router(api_router)

app.add_exception_handler(ServiceException, ExceptionHandlers.service_exception_handler)
app.add_exception_handler(StarletteHTTPException, ExceptionHandlers.http_exception_handler)
app.add_exception_handler(Exception, ExceptionHandlers.unhandled_exception_handler)
