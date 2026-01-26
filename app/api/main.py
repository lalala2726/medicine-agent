from fastapi import APIRouter

from app.api.routes.image import router as image_router

api_router = APIRouter(prefix="/api")
api_router.include_router(image_router)
