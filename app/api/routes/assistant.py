import json
from typing import AsyncIterable

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from agent.tools.user_tool import get_user_info
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
    if not request.question:
        raise ServiceException(code=ResponseCode.BAD_REQUEST, message="问题不能为空")

    model = create_chat_model()

    # 创建支持流式的 Agent
    agent = create_agent(
        model,
        tools=[get_user_info],
        system_prompt="You are a helpful assistant.",
    )

    async def event_stream() -> AsyncIterable[str]:
        """SSE 事件流生成器"""
        # 使用 astream 并开启 messages 模式以获取每个 token
        stream = agent.astream(
            input={"messages": [HumanMessage(content=request.question)]},
            stream_mode="messages"
        )

        async for chunk, metadata in stream:
            # 在 messages 模式下，chunk 是消息对象（如 AIMessageChunk）
            if chunk.content:
                payload = AssistantResponse(content=chunk.content, is_end=False)
                yield f"data: {json.dumps(payload.model_dump(), ensure_ascii=False)}\n\n"

        # 发送结束标识
        payload = AssistantResponse(content="", is_end=True)
        yield f"data: {json.dumps(payload.model_dump(), ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")