import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.tools.user_tool import get_user_info
from app.core.codes import ResponseCode
from app.core.exceptions import ServiceException
from app.core.llm import create_chat_model

router = APIRouter(prefix="/assistant", tags=["AI助手"])


class AssistantRequest(BaseModel):
    """AI助手请求参数"""

    question: str = Field(..., description="问题")


class AssistantResponse(BaseModel):
    """AI助手响应参数"""
    content: str = Field(..., description="答案")
    is_end: bool = Field(default=False, description="是否结束")


@router.post("/chat", summary="AI助手对话")
async def assistant(request: AssistantRequest) -> StreamingResponse:
    """
    AI 助手对话接口，SSE 流式输出

    Args:
        request: AI助手请求参数

    Returns:
        StreamingResponse: SSE 流式响应

    Raises:
        ServiceException: 问题为空时抛出异常
    """
    if not request.question:
        raise ServiceException(code=ResponseCode.BAD_REQUEST, message="问题不能为空")

    model = create_chat_model().bind_tools([get_user_info])

    async def event_stream():
        """SSE 事件流生成器"""
        async for chunk in model.astream(request.question):
            if chunk.content:
                payload = AssistantResponse(content=chunk.content, is_end=False)
                yield f"data: {json.dumps(payload.model_dump())}\n\n"
        payload = AssistantResponse(content="", is_end=True)
        yield f"data: {json.dumps(payload.model_dump())}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
