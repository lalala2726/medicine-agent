from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from app.schemas.base_request import PageRequest
from app.schemas.response import ApiResponse, PageResponse
from app.services.admin_assisant_service import assistant_chat, chat_list

router = APIRouter(prefix="/admin/assistant", tags=["智能助手"])


class AssistantRequest(BaseModel):
    """AI助手请求参数。"""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1, description="问题")
    conversation_uuid: str | None = Field(
        default=None,
        min_length=1,
        description="会话UUID",
    )


class AssistantChatListRequest(BaseModel):
    """智能助手会话列表请求参数。"""
    model_config = ConfigDict(extra="forbid")

    page_num: int = Field(default=1, ge=1, description="页号")
    page_size: int = Field(default=20, ge=1, le=100, description="每页大小")


@router.post("/chat", summary="普通对话")
async def assistant(request: AssistantRequest) -> StreamingResponse:
    """管理助手聊天接口（SSE 流式返回）。"""

    return assistant_chat(
        question=request.question,
        conversation_uuid=request.conversation_uuid,
    )


@router.get("/chat/list", summary="管理助手会话列表")
async def assistant_chat_list(
        request: AssistantChatListRequest = Depends(),
) -> ApiResponse[PageResponse[dict[str, str]]]:
    """
    分页查询管理助手会话列表（仅返回会话 UUID 与标题）。
    """

    rows, total = chat_list(
        page_request=PageRequest(
            page_num=request.page_num,
            page_size=request.page_size,
        )
    )
    return ApiResponse.page(
        rows=rows,
        total=total,
        page_num=request.page_num,
        page_size=request.page_size,
    )
