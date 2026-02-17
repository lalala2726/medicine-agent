from fastapi import APIRouter

from app.core.request_context import get_current_token
from app.schemas.response import ApiResponse
from app.services.user_service import get_current_user, get_current_user_id

router = APIRouter(prefix="/test", tags=["测试"])


@router.get("/test1")
async def test() -> ApiResponse[dict]:
    user = get_current_user()
    return ApiResponse.success(
        data={
            "user": user.model_dump(exclude_none=True),
            "user_id": get_current_user_id(),
            "token": get_current_token(),
        }
    )
