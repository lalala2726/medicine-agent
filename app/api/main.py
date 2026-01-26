from fastapi import APIRouter

from app.api.routes.image_parse import router as image_router
from app.api.routes.assistant import router as assistant_router


api_router = APIRouter(prefix="/api")
api_router.include_router(image_router)
api_router.include_router(assistant_router)
