import json
from typing import AsyncIterable

from fastapi.concurrency import run_in_threadpool
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.admin.workflow import build_graph
from app.core.codes import ResponseCode
from app.core.exceptions import ServiceException
from app.core.langsmith import build_langsmith_runnable_config

router = APIRouter(prefix="/admin/assistant", tags=["管理助手"])
ADMIN_WORKFLOW = build_graph()
STREAM_OUTPUT_NODES = {"order_agent", "chat_agent"}


class AssistantRequest(BaseModel):
    """AI助手请求参数"""
    question: str = Field(..., description="问题")


class AssistantResponse(BaseModel):
    """AI助手响应参数"""
    content: str = Field(..., description="答案")
    is_end: bool = Field(default=False, description="是否结束")


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


def _message_chunk_to_text(message_chunk) -> str:
    content = getattr(message_chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                part = item.get("text")
                if isinstance(part, str):
                    text_parts.append(part)
            elif isinstance(item, str):
                text_parts.append(item)
        return "".join(text_parts)
    if content is None:
        return ""
    return str(content)


def _has_plan(state: dict) -> bool:
    raw_plan = state.get("plan")
    return isinstance(raw_plan, list) and len(raw_plan) > 0


def _should_stream_node_by_state(state: dict, node_name: str) -> bool:
    if node_name == "chat_agent":
        return True

    if node_name != "order_agent":
        return False

    routing = state.get("routing") or {}
    route_target = routing.get("route_target")
    if route_target == "order_agent" and not _has_plan(state):
        return True

    next_nodes = routing.get("next_nodes")
    return (
        bool(routing.get("is_final_stage"))
        and isinstance(next_nodes, list)
        and len(next_nodes) == 1
        and next_nodes[0] == "order_agent"
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
            latest_state = state
            has_streamed_output = False

            if hasattr(ADMIN_WORKFLOW, "astream"):
                config = _build_stream_config()
                stream_kwargs = {
                    "stream_mode": ["messages", "values"],
                }
                if config:
                    stream_kwargs["config"] = config

                async for mode, chunk in ADMIN_WORKFLOW.astream(state, **stream_kwargs):
                    if mode == "messages":
                        message_chunk, metadata = chunk
                        stream_node = metadata.get("langgraph_node")
                        if stream_node in STREAM_OUTPUT_NODES and _should_stream_node_by_state(latest_state, stream_node):
                            token_text = _message_chunk_to_text(message_chunk)
                            if token_text:
                                has_streamed_output = True
                                yield _build_payload(token_text, False)
                    elif mode == "values" and isinstance(chunk, dict):
                        latest_state = chunk
            else:
                latest_state = await run_in_threadpool(_invoke_admin_workflow, state)

            if not has_streamed_output:
                yield _build_payload(_extract_content(latest_state), False)
        except ServiceException as exc:
            yield _build_payload(f"处理失败: {exc.message}", False)
        except Exception:
            yield _build_payload("服务暂时不可用，请稍后重试。", False)
        finally:
            yield _build_payload("", True)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
