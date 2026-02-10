import asyncio
import json
from contextlib import suppress
from typing import Any, AsyncIterable

from fastapi.concurrency import run_in_threadpool
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.utils.streaming_utils import extract_text, is_final_node
from app.agent.admin.workflow import build_graph
from app.core.assistant_status import reset_status_emitter, set_status_emitter
from app.core.codes import ResponseCode
from app.core.exceptions import ServiceException
from app.core.langsmith import build_langsmith_runnable_config
from app.schemas.sse_response import AssistantResponse, Content, MessageType

router = APIRouter(prefix="/admin/assistant", tags=["管理助手"])
ADMIN_WORKFLOW = build_graph()
STREAM_OUTPUT_NODES = {"order_agent", "chat_agent", "summary_agent"}


class AssistantRequest(BaseModel):
    """AI助手请求参数"""

    question: str = Field(..., description="问题")


def _invoke_admin_workflow(state: dict) -> dict:
    config = build_langsmith_runnable_config(
        run_name="admin_assistant_graph",
        tags=["admin-assistant", "langgraph"],
        metadata={"entrypoint": "api.admin_assistant.chat"},
    )
    if config:
        return ADMIN_WORKFLOW.invoke(state, config=config)
    return ADMIN_WORKFLOW.invoke(state)


def _build_stream_config() -> dict | None:
    return build_langsmith_runnable_config(
        run_name="admin_assistant_graph",
        tags=["admin-assistant", "langgraph"],
        metadata={"entrypoint": "api.admin_assistant.chat"},
    )


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

        summary_result = results.get("summary") or {}
        summary_content = summary_result.get("content")
        if isinstance(summary_content, str) and summary_content:
            return summary_content

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

        def _to_sse(payload: AssistantResponse) -> str:
            return f"data: {json.dumps(payload.model_dump(mode='json', exclude_none=True), ensure_ascii=False)}\n\n"

        def _build_answer_payload(content: str, is_end: bool) -> str:
            payload = AssistantResponse(
                content=Content(text=content),
                type=MessageType.ANSWER,
                is_end=is_end,
            )
            return _to_sse(payload)

        def _build_status_payload(status_content: dict[str, Any]) -> str:
            payload = AssistantResponse(
                content=Content(
                    text=status_content.get("text"),
                    node=status_content.get("node"),
                    state=status_content.get("state"),
                    message=status_content.get("message"),
                    result=status_content.get("result"),
                    name=status_content.get("name"),
                    arguments=status_content.get("arguments"),
                ),
                type=MessageType.STATUS,
            )
            return _to_sse(payload)

        def _emit_error_payload(exc: Exception) -> str:
            if isinstance(exc, ServiceException):
                return _build_answer_payload(f"处理失败: {exc.message}", False)
            return _build_answer_payload("服务暂时不可用，请稍后重试。", False)

        state = _build_initial_state(request.question)
        latest_state = state
        has_streamed_output = False
        has_emitted_error = False

        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _status_emitter(event: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, ("status", event))

        async def _produce_workflow_events() -> None:
            nonlocal latest_state
            try:
                if hasattr(ADMIN_WORKFLOW, "astream"):
                    config = _build_stream_config()
                    stream_kwargs = {
                        "stream_mode": ["messages", "values"],
                    }
                    if config:
                        stream_kwargs["config"] = config

                    async for mode, chunk in ADMIN_WORKFLOW.astream(state, **stream_kwargs):
                        await queue.put(("graph", (mode, chunk)))
                else:
                    latest_state = await run_in_threadpool(_invoke_admin_workflow, state)
            except Exception as exc:
                await queue.put(("error", exc))
            finally:
                await queue.put(("done", None))

        emitter_token = set_status_emitter(_status_emitter)
        producer_task = asyncio.create_task(_produce_workflow_events())

        try:
            while True:
                event_type, payload = await queue.get()

                if event_type == "status":
                    yield _build_status_payload(payload)
                    continue

                if event_type == "graph":
                    mode, chunk = payload
                    if mode == "messages":
                        message_chunk, metadata = chunk
                        stream_node = metadata.get("langgraph_node")
                        if stream_node in STREAM_OUTPUT_NODES and (
                            stream_node == "chat_agent"
                            or is_final_node(latest_state, stream_node)
                        ):
                            token_text = extract_text(message_chunk)
                            if token_text:
                                has_streamed_output = True
                                yield _build_answer_payload(token_text, False)
                    elif mode == "values" and isinstance(chunk, dict):
                        latest_state = chunk
                    continue

                if event_type == "error":
                    has_emitted_error = True
                    yield _emit_error_payload(payload)
                    continue

                if event_type == "done":
                    # 消费 done 前后可能并发压入的尾部状态事件
                    await asyncio.sleep(0)
                    while True:
                        try:
                            pending_type, pending_payload = queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break

                        if pending_type == "status":
                            yield _build_status_payload(pending_payload)
                            continue
                        if pending_type == "graph":
                            pending_mode, pending_chunk = pending_payload
                            if pending_mode == "messages":
                                message_chunk, metadata = pending_chunk
                                stream_node = metadata.get("langgraph_node")
                                if stream_node in STREAM_OUTPUT_NODES and (
                                    stream_node == "chat_agent"
                                    or is_final_node(latest_state, stream_node)
                                ):
                                    token_text = extract_text(message_chunk)
                                    if token_text:
                                        has_streamed_output = True
                                        yield _build_answer_payload(token_text, False)
                            elif pending_mode == "values" and isinstance(pending_chunk, dict):
                                latest_state = pending_chunk
                            continue
                        if pending_type == "error":
                            has_emitted_error = True
                            yield _emit_error_payload(pending_payload)

                    break

            if not has_emitted_error and not has_streamed_output:
                yield _build_answer_payload(_extract_content(latest_state), False)
        finally:
            reset_status_emitter(emitter_token)
            if not producer_task.done():
                producer_task.cancel()
                with suppress(asyncio.CancelledError):
                    await producer_task
            yield _build_answer_payload("", True)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
