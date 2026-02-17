from fastapi import APIRouter

from services.user_service import get_user_info

router = APIRouter(prefix="/test", tags=["测试"])

@router.get("/test1")
async def test() -> None:
    await get_user_info()
