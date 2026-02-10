from fastapi import APIRouter

from app.api.routes.admin_assistant import router as assistant_router
from app.api.routes.image_parse import router as image_router
from app.api.routes.knowledge_base import router as knowledge_base_router

api_router = APIRouter(prefix="/api")
api_router.include_router(image_router)
api_router.include_router(assistant_router)
api_router.include_router(knowledge_base_router)
