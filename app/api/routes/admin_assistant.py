import json
from typing import AsyncIterable

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.admin_agent import stream_admin_assistant
from app.core.codes import ResponseCode
from app.core.exceptions import ServiceException

router = APIRouter(prefix="/admin/assistant", tags=["管理助手"])


class AssistantRequest(BaseModel):
    """AI助手请求参数"""
    question: str = Field(..., description="问题")


class AssistantResponse(BaseModel):
    """AI助手响应参数"""
    content: str = Field(..., description="答案")
    is_end: bool = Field(default=False, description="是否结束")


@router.post("/chat", summary="管理助手对话")
async def assistant(request: AssistantRequest) -> StreamingResponse:
    if not request.question:
        raise ServiceException(code=ResponseCode.BAD_REQUEST, message="问题不能为空")

    async def event_stream() -> AsyncIterable[str]:
        """SSE 事件流生成器"""

        def _build_payload(content: str, is_end: bool) -> str:
            payload = AssistantResponse(content=content, is_end=is_end)
            return f"data: {json.dumps(payload.model_dump(), ensure_ascii=False)}\n\n"

        async for content, is_end in stream_admin_assistant(request.question):
            yield _build_payload(content, is_end)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
