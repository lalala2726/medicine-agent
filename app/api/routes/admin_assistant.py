import json
from typing import AsyncIterable

from fastapi.concurrency import run_in_threadpool
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.admin.workflow import build_graph
from app.core.codes import ResponseCode
from app.core.exceptions import ServiceException

router = APIRouter(prefix="/admin/assistant", tags=["管理助手"])
ADMIN_WORKFLOW = build_graph()


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

    def _build_initial_state(question: str) -> dict:
        return {
            "user_input": question,
            "user_intent": {},
            "plan": [],
            "routing": {},
            "order_context": {},
            "aftersale_context": {},
            "excel_context": {},
            "shared_memory": {},
            "results": {},
            "errors": [],
        }

    def _extract_content(final_state: dict) -> str:
        results = final_state.get("results") or {}
        chat_result = results.get("chat") or {}
        chat_content = chat_result.get("content")
        if isinstance(chat_content, str) and chat_content:
            return chat_content

        order_context = final_state.get("order_context") or {}
        order_result = order_context.get("result") or {}
        order_content = order_result.get("content")
        if isinstance(order_content, str) and order_content:
            return order_content

        errors = final_state.get("errors") or []
        if errors:
            return "；".join(str(item) for item in errors)

        if results:
            return json.dumps(results, ensure_ascii=False)

        return "已完成处理。"

    async def event_stream() -> AsyncIterable[str]:
        """SSE 事件流生成器"""

        def _build_payload(content: str, is_end: bool) -> str:
            payload = AssistantResponse(content=content, is_end=is_end)
            return f"data: {json.dumps(payload.model_dump(), ensure_ascii=False)}\n\n"

        try:
            state = _build_initial_state(request.question)
            final_state = await run_in_threadpool(ADMIN_WORKFLOW.invoke, state)
            yield _build_payload(_extract_content(final_state), False)
        except ServiceException as exc:
            yield _build_payload(f"处理失败: {exc.message}", False)
        except Exception:
            yield _build_payload("服务暂时不可用，请稍后重试。", False)
        finally:
            yield _build_payload("", True)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
