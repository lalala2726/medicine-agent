import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

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
    is_end: bool = False


@router.post("/chat", summary="AI助手")
async def assistant(request: AssistantRequest) -> StreamingResponse:
    if not request.question:
        raise ServiceException(code=ResponseCode.BAD_REQUEST, message="问题不能为空")

    model = create_chat_model()

    async def event_stream():
        async for chunk in model.astream(request.question):
            if chunk.content:
                # 将大模型返回的切片信息放在响应里面，然后立即包装为SSE方式然后返回
                payload = AssistantResponse(content=chunk.content, is_end=False)
                yield f"data: {json.dumps(payload.model_dump())}\n\n"
        payload = AssistantResponse(content="", is_end=True)
        yield f"data: {json.dumps(payload.model_dump())}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
