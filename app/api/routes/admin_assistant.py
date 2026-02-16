from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from app.services.admin_assisant_service import assistant_chat

router = APIRouter(prefix="/admin/assistant", tags=["管理助手"])


class AssistantRequest(BaseModel):
    """AI助手请求参数。"""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1, description="问题")
    conversation_uuid: str | None = Field(
        default=None,
        min_length=1,
        description="会话UUID",
    )


@router.post("/chat", summary="管理助手对话")
async def assistant(request: AssistantRequest) -> StreamingResponse:
    """管理助手聊天接口（SSE 流式返回）。"""

    return assistant_chat(
        question=request.question,
        conversation_uuid=request.conversation_uuid,
    )
