import json
from typing import AsyncIterable

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field

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


class AgentInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    messages: list[HumanMessage]


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
        def _build_payload(content: str, is_end: bool) -> str:
            payload = AssistantResponse(content=content, is_end=is_end)
            return f"data: {json.dumps(payload.model_dump(), ensure_ascii=False)}\n\n"

        try:
            # 使用 astream 并开启 messages 模式以获取每个 token
            agent_input = AgentInput(messages=[HumanMessage(content=request.question)])
            stream = agent.astream(input=agent_input, stream_mode="messages")

            async for chunk, metadata in stream:
                # 在 messages 模式下，chunk 是消息对象（如 AIMessageChunk）
                if chunk.content:
                    yield _build_payload(chunk.content, False)
        except ServiceException as exc:
            if exc.code == ResponseCode.UNAUTHORIZED:
                message = exc.message
            else:
                message = f"处理失败: {exc.message}"
            yield _build_payload(message, True)
            return
        except Exception:
            yield _build_payload("服务暂时不可用，请稍后重试。", True)
            return

        # 发送结束标识
        yield _build_payload("", True)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
