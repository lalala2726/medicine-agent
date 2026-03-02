from __future__ import annotations

import asyncio
import threading
import uuid
from dataclasses import dataclass
from typing import Any

from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from loguru import logger

from app.agent.assistant.state import ChatHistoryMessage, ExecutionTraceState, TokenUsageState
from app.agent.assistant.workflow import build_graph
from app.core.agent.agent_orchestrator import (
    AssistantStreamConfig,
    create_streaming_response,
)
from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException
from app.core.langsmith import build_langsmith_runnable_config
from app.core.llms import create_chat_model
from app.core.security.auth_context import get_user_id
from app.core.speech import build_message_tts_stream
from app.schemas.admin_assistant_history import ConversationMessageResponse
from app.schemas.base_request import PageRequest
from app.schemas.document.conversation import ConversationListItem
from app.schemas.document.message import MessageRole, MessageStatus
from app.schemas.sse_response import AssistantResponse, Content, MessageType
from app.services.conversation_service import (
    add_admin_conversation,
    delete_admin_conversation,
    get_admin_conversation,
    list_admin_conversations,
    save_conversation_title,
    update_admin_conversation_title,
)
from app.services.message_service import add_message, list_messages
from app.services.message_trace_service import add_message_trace
from app.services.token_usage_service import (
    resolve_persistable_token_usage,
    resolve_persistable_trace_token_usage,
)
from app.utils.prompt_utils import load_prompt

ADMIN_WORKFLOW = build_graph()
STREAM_OUTPUT_NODES = {
    "chat_agent",
    "order_agent",
    "product_agent",
    "after_sale_agent",
    "user_agent",
    "analytics_agent",
    "adaptive_agent",
}
EMPTY_ASSISTANT_ANSWER_FALLBACK = "服务暂时不可用，请稍后重试。"


@dataclass(frozen=True)
class ConversationContext:
    """
    会话准备阶段的统一上下文。

    Attributes:
        conversation_uuid: 会话 UUID。
        conversation_id: Mongo 会话 ID（ObjectId 字符串）。
        assistant_message_uuid: 本轮 AI 回复消息 UUID（在流开始前预生成）。
        history_messages: 会话历史消息（按时间正序）。
        initial_emitted_events: 流开始前要注入的 SSE 事件。
        is_new_conversation: 是否为本次请求创建的新会话。
    """

    conversation_uuid: str
    conversation_id: str
    assistant_message_uuid: str
    history_messages: list[ChatHistoryMessage]
    initial_emitted_events: tuple[AssistantResponse, ...]
    is_new_conversation: bool


def _invoke_admin_workflow(state: dict[str, Any]) -> dict[str, Any]:
    """
    同步执行管理助手 workflow。

    该函数用于 `astream` 不可用时的回退路径，保证接口始终可返回结果。
    """

    config = build_langsmith_runnable_config(
        run_name="admin_assistant_graph",
        tags=["admin-assistant", "langgraph"],
        metadata={"entrypoint": "api.admin_assistant.chat"},
    )
    if config:
        return ADMIN_WORKFLOW.invoke(state, config=config)
    return ADMIN_WORKFLOW.invoke(state)


def _build_stream_config() -> dict | None:
    """
    构建流式执行使用的 LangSmith 配置。

    返回 None 时表示不启用额外 tracing 配置。
    """

    return build_langsmith_runnable_config(
        run_name="admin_assistant_graph",
        tags=["admin-assistant", "langgraph"],
        metadata={"entrypoint": "api.admin_assistant.chat"},
    )


def _build_initial_state(
        question: str,
        *,
        history_messages: list[ChatHistoryMessage] | None = None,
        enable_thinking: bool = False,
) -> dict[str, Any]:
    """
    构造管理助手的初始状态。

    所有节点共享该状态结构，避免执行过程中出现缺失键导致的分支判断复杂化。

    Args:
        question: 当前用户问题文本；当前实现仅用于接口语义占位，不参与状态构造。
        history_messages: 会话历史消息列表；为空时默认空列表。
        enable_thinking: 是否开启深度思考透传；默认 `False`。

    Returns:
        dict[str, Any]: LangGraph 初始状态字典，包含路由、历史、消息与透传开关等字段。
    """

    _ = question
    base_history = list(history_messages or [])

    return {
        "routing": {
            "route_targets": [],
            "task_difficulty": "normal",
        },
        "context": "",
        "history_messages": base_history,
        "enable_thinking": bool(enable_thinking),
        "execution_traces": [],
        "token_usage": None,
        "result": "",
        # 兼容 MessagesState，保证 astream(messages) 能消费到上下文消息。
        "messages": list(base_history),
    }


def _should_stream_token(stream_node: str | None, latest_state: dict[str, Any]) -> bool:
    """
    判定某个节点 token 是否应该被推送给前端。

    当前规则允许所有业务输出节点输出 token（gateway 节点除外）。
    """

    _ = latest_state
    return stream_node in STREAM_OUTPUT_NODES


def _map_exception(exc: Exception) -> str:
    """
    将内部异常映射为对用户友好的文案。

    业务异常会保留明确错误信息，未知异常统一为通用降级提示。
    """

    if isinstance(exc, ServiceException):
        return f"处理失败: {exc.message}"
    return EMPTY_ASSISTANT_ANSWER_FALLBACK


def _build_conversation_created_event(
        *,
        conversation_uuid: str,
        message_uuid: str,
) -> AssistantResponse:
    """
    构造“会话创建成功”的前置 SSE 事件。

    该事件会在流开始后优先发送给前端，便于前端立即拿到会话标识。
    """

    return AssistantResponse(
        content=Content(
            state="created",
            message="会话创建成功",
        ),
        type=MessageType.NOTICE,
        meta={
            "conversation_uuid": conversation_uuid,
            "message_uuid": message_uuid,
        },
    )


def _build_message_prepared_event(
        *,
        message_uuid: str,
) -> AssistantResponse:
    """
    构造“消息已创建”的前置 SSE 事件。

    该事件用于旧会话场景，前端可在任何 answer token 之前先拿到本轮 AI 消息 UUID。
    """

    return AssistantResponse(
        type=MessageType.NOTICE,
        meta={
            "message_uuid": message_uuid,
        },
    )


def _generate_and_save_title(*, conversation_uuid: str, question: str) -> None:
    """
    生成并持久化会话标题。

    该函数运行在后台线程中，避免阻塞 SSE 主链路。
    """

    try:
        title = generate_title(question).strip() or "未知标题"
        save_conversation_title(
            conversation_uuid=conversation_uuid,
            title=title,
        )
    except Exception as exc:  # pragma: no cover - 防御性兜底
        logger.opt(exception=exc).warning(
            "Failed to generate/save conversation title conversation_uuid={conversation_uuid}",
            conversation_uuid=conversation_uuid,
        )


def _schedule_title_generation(*, conversation_uuid: str, question: str) -> None:
    """
    并行调度标题生成任务。

    优先使用当前事件循环把阻塞任务放入线程池；若当前无事件循环，
    则退化为守护线程执行（兼容同步调用/单元测试场景）。
    """

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        thread = threading.Thread(
            target=_generate_and_save_title,
            kwargs={
                "conversation_uuid": conversation_uuid,
                "question": question,
            },
            daemon=True,
        )
        thread.start()
        return

    loop.create_task(
        asyncio.to_thread(
            _generate_and_save_title,
            conversation_uuid=conversation_uuid,
            question=question,
        )
    )


def _load_admin_conversation(
        *,
        conversation_uuid: str,
        user_id: int,
) -> str:
    """
    加载会话并返回 Mongo 会话ID（ObjectId 字符串）。

    Raises:
        ServiceException: 会话不存在、无权限或会话数据异常时抛出。
    """

    conversation = get_admin_conversation(
        conversation_uuid=conversation_uuid,
        user_id=user_id,
    )
    if conversation is None:
        raise ServiceException(code=ResponseCode.NOT_FOUND, message="会话不存在")

    conversation_id = conversation.id
    if conversation_id is None:
        raise ServiceException(code=ResponseCode.DATABASE_ERROR, message="会话数据异常")
    return str(conversation_id)


def _schedule_background_task(
        *,
        task_name: str,
        func: Any,
        kwargs: dict[str, Any],
) -> None:
    """
    使用守护线程调度后台任务。

    Args:
        task_name: 任务名称（用于日志追踪）。
        func: 要执行的函数。
        kwargs: 关键字参数。

    Returns:
        None
    """

    def _runner() -> None:
        try:
            func(**kwargs)
        except Exception as exc:  # pragma: no cover - 防御性兜底
            logger.opt(exception=exc).warning("Background task failed task_name={task_name}", task_name=task_name)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()


def _persist_user_message(
        *,
        conversation_id: str,
        question: str,
) -> None:
    """
    后台持久化 user 消息。

    Args:
        conversation_id: 会话 ID。
        question: 用户问题文本。

    Returns:
        None
    """

    add_message(conversation_id=conversation_id, role="user", content=question)


def _persist_assistant_message(
        *,
        conversation_id: str,
        message_uuid: str,
        answer_text: str,
        execution_trace: list[ExecutionTraceState] | None,
        token_usage: TokenUsageState | dict[str, Any] | None,
        status: MessageStatus | str,
        thinking_text: str | None = None,
) -> None:
    """
    持久化 ai 消息。

    流程：
    1. 主消息表保存基础消息 + token 总量；
    2. trace 表保存 workflow 汇总 + execution_trace + token 汇总；
    3. trace 失败仅记录 warning，不影响主消息保存。
    """
    resolved_status = MessageStatus(status)
    persistable_token_usage = resolve_persistable_token_usage(
        token_usage,
        execution_trace,
    )
    persistable_trace_token_usage = resolve_persistable_trace_token_usage(
        token_usage,
        execution_trace,
    )
    add_message(
        conversation_id=conversation_id,
        role="ai",
        status=resolved_status,
        content=answer_text,
        thinking=thinking_text,
        token_usage=persistable_token_usage,
        message_uuid=message_uuid,
    )
    try:
        add_message_trace(
            message_uuid=message_uuid,
            conversation_id=conversation_id,
            execution_trace=execution_trace,
            token_usage=persistable_trace_token_usage,
            has_error=resolved_status == MessageStatus.ERROR,
        )
    except Exception as exc:  # pragma: no cover - 防御性兜底
        logger.opt(exception=exc).warning(
            "Persist message_trace failed message_uuid={message_uuid}",
            message_uuid=message_uuid,
        )


def _build_assistant_message_callback(*, conversation_id: str, assistant_message_uuid: str):
    """
    构建“流结束后写入 AI 消息”的异步回调。

    该回调由流式引擎在输出结束时触发，满足“AI 完整响应后再落库”的时序要求。
    """

    async def _callback(
            answer_text: str,
            execution_trace: list[ExecutionTraceState] | None,
            token_usage: TokenUsageState | dict[str, Any] | None = None,
            has_error: bool = False,
            thinking_text: str | None = None,
    ) -> None:
        resolved_token_usage = token_usage
        resolved_has_error = has_error
        # 兼容旧回调签名：(answer_text, execution_trace, has_error)
        if isinstance(token_usage, bool):
            resolved_token_usage = None
            resolved_has_error = token_usage

        normalized_answer = str(answer_text or "").strip()
        resolved_status = MessageStatus.ERROR if resolved_has_error else MessageStatus.SUCCESS
        if not normalized_answer:
            normalized_answer = EMPTY_ASSISTANT_ANSWER_FALLBACK
            resolved_status = MessageStatus.ERROR

        _schedule_background_task(
            task_name="persist_assistant_message",
            func=_persist_assistant_message,
            kwargs={
                "conversation_id": conversation_id,
                "message_uuid": assistant_message_uuid,
                "answer_text": normalized_answer,
                "thinking_text": thinking_text,
                "execution_trace": execution_trace,
                "token_usage": resolved_token_usage,
                "status": resolved_status,
            },
        )

    return _callback


def _prepare_new_conversation(
        *,
        question: str,
        user_id: int,
        assistant_message_uuid: str,
) -> ConversationContext:
    """
    准备新会话上下文。

    Args:
        question: 当前用户问题文本，用于异步生成标题。
        user_id: 当前用户 ID。

    Returns:
        ConversationContext: 新会话上下文（含会话创建事件）。

    Raises:
        ServiceException: 当会话创建失败时抛出数据库异常。
    """

    conversation_uuid = str(uuid.uuid4())
    conversation_id = add_admin_conversation(
        conversation_uuid=conversation_uuid,
        user_id=user_id,
    )
    if conversation_id is None:
        raise ServiceException(code=ResponseCode.DATABASE_ERROR, message="无法创建会话，请稍后重试。")

    _schedule_title_generation(
        conversation_uuid=conversation_uuid,
        question=question,
    )
    return ConversationContext(
        conversation_uuid=conversation_uuid,
        conversation_id=conversation_id,
        assistant_message_uuid=assistant_message_uuid,
        history_messages=[HumanMessage(content=question)],
        initial_emitted_events=(
            _build_conversation_created_event(
                conversation_uuid=conversation_uuid,
                message_uuid=assistant_message_uuid,
            ),
        ),
        is_new_conversation=True,
    )


def _prepare_existing_conversation(
        *,
        conversation_uuid: str,
        user_id: int,
        question: str,
        assistant_message_uuid: str,
) -> ConversationContext:
    """
    准备已存在会话上下文。

    Args:
        conversation_uuid: 会话 UUID。
        user_id: 当前用户 ID（用于会话归属校验）。
        question: 当前用户问题文本，会追加到历史末尾供模型推理。

    Returns:
        ConversationContext: 已存在会话上下文（含消息预创建事件），
            其中 `history_messages` 末尾包含本轮用户问题。

    Raises:
        ServiceException: 当会话不存在、无权限或数据库异常时抛出。
    """

    conversation_id = _load_admin_conversation(
        conversation_uuid=conversation_uuid,
        user_id=user_id,
    )
    history_messages = load_history(
        conversation_id=conversation_id,
        limit=50,
    )
    # 旧会话场景需显式注入本轮用户输入，避免模型只基于上一轮历史作答。
    history_messages = [*history_messages, HumanMessage(content=question)]
    return ConversationContext(
        conversation_uuid=conversation_uuid,
        conversation_id=conversation_id,
        assistant_message_uuid=assistant_message_uuid,
        history_messages=history_messages,
        initial_emitted_events=(
            _build_message_prepared_event(
                message_uuid=assistant_message_uuid,
            ),
        ),
        is_new_conversation=False,
    )


def _prepare_conversation_context(
        *,
        question: str,
        user_id: int,
        conversation_uuid: str | None,
        assistant_message_uuid: str,
) -> ConversationContext:
    """
    统一准备会话上下文（新建/加载分支）。

    Args:
        question: 当前用户问题文本。
        user_id: 当前用户 ID。
        conversation_uuid: 会话 UUID；为空时创建新会话。

    Returns:
        ConversationContext: 可直接驱动后续流式响应的会话上下文。

    Raises:
        ServiceException: 当会话创建/加载失败时抛出。
    """

    if conversation_uuid is None:
        return _prepare_new_conversation(
            question=question,
            user_id=user_id,
            assistant_message_uuid=assistant_message_uuid,
        )
    return _prepare_existing_conversation(
        conversation_uuid=conversation_uuid,
        user_id=user_id,
        question=question,
        assistant_message_uuid=assistant_message_uuid,
    )


def assistant_chat(
        *,
        question: str,
        conversation_uuid: str | None = None,
        enable_thinking: bool = False,
) -> StreamingResponse:
    """
    管理助手聊天入口（SSE 流式返回）。

    行为说明：
    1. 每次请求都会先预生成本轮 AI `message_uuid`；
    2. 首次消息（未传 `conversation_uuid`）会创建会话并先推送“会话创建成功”事件；
    3. 旧会话会先推送“消息已创建”事件，确保前端先拿到本轮 `message_uuid`；
    4. 用户提问会后台写入消息表；
    5. AI 回复会在流结束后后台写入消息表（空输出也会落库 error 兜底文案）；
    6. 标题生成与保存在后台并行执行，不阻塞当前流式响应。
    7. 会话准备逻辑由 `_prepare_conversation_context` 统一处理。

    Args:
        question: 用户输入问题文本。
        conversation_uuid: 可选会话 UUID；为空时创建新会话。
        enable_thinking: 是否开启深度思考流式透传；默认 `False`。

    Returns:
        StreamingResponse: 标准 SSE 流式响应对象。
    """

    current_user_id = get_user_id()
    assistant_message_uuid = str(uuid.uuid4())
    context = _prepare_conversation_context(
        question=question,
        user_id=current_user_id,
        conversation_uuid=conversation_uuid,
        assistant_message_uuid=assistant_message_uuid,
    )

    _schedule_background_task(
        task_name="persist_user_message",
        func=_persist_user_message,
        kwargs={
            "conversation_id": context.conversation_id,
            "question": question,
        },
    )

    stream_config = AssistantStreamConfig(
        workflow=ADMIN_WORKFLOW,
        build_initial_state=lambda q: _build_initial_state(
            q,
            history_messages=context.history_messages,
            enable_thinking=enable_thinking,
        ),
        extract_final_content=lambda state: str(state.get("result") or ""),
        should_stream_token=_should_stream_token,
        build_stream_config=_build_stream_config,
        invoke_sync=_invoke_admin_workflow,
        map_exception=_map_exception,
        on_answer_completed=_build_assistant_message_callback(
            conversation_id=context.conversation_id,
            assistant_message_uuid=context.assistant_message_uuid,
        ),
        initial_emitted_events=context.initial_emitted_events,
    )
    return create_streaming_response(question, stream_config)


def assistant_message_tts_stream(
        *,
        message_uuid: str,
) -> StreamingResponse:
    """
    管理助手消息转语音（HTTP chunked audio stream）。

    说明：
    - 先基于 `message_uuid` 做消息存在性、会话归属、消息角色校验；
    - 校验通过后建立上游 Volcengine 双向 TTS websocket；
    - 下游以音频字节流（chunked）持续返回给前端。
    """

    current_user_id = get_user_id()
    tts_stream = build_message_tts_stream(
        message_uuid=message_uuid,
        user_id=current_user_id,
    )
    return StreamingResponse(
        tts_stream.audio_stream,
        media_type=tts_stream.media_type,
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def load_history(
        *,
        conversation_id: str,
        limit: int = 50,
) -> list[ChatHistoryMessage]:
    """
    加载会话历史消息（最近窗口）。

    读取策略：
    1. 先按创建时间倒序读取最近 N 条；
    2. 再反转为正序，保证喂给模型时上下文顺序正确。
    """

    message_documents = list_messages(
        conversation_id=conversation_id,
        limit=limit,
        ascending=False,
    )
    history_messages: list[ChatHistoryMessage] = [
        HumanMessage(content=document.content)
        if document.role == MessageRole.USER
        else AIMessage(content=document.content)
        for document in message_documents
    ]
    return list(reversed(history_messages))


def conversation_list(
        *,
        page_request: PageRequest,
) -> tuple[list[ConversationListItem], int]:
    """
    分页查询当前用户的管理助手会话列表。

    仅返回会话 UUID 与标题，不返回会话内部主键等冗余字段。

    Args:
        page_request: 分页请求参数。

    Returns:
        tuple[list[ConversationListItem], int]:
            - rows: 当前页会话列表项模型。
            - total: 会话总数。
    """

    current_user_id = get_user_id()
    return list_admin_conversations(
        user_id=current_user_id,
        page_num=page_request.page_num,
        page_size=page_request.page_size,
    )


def conversation_messages(
        *,
        conversation_uuid: str,
        page_request: PageRequest,
) -> list[ConversationMessageResponse]:
    """
    分页查询当前用户某个管理助手会话的历史消息。

    分页规则：
    1. page_num=1 返回最近一页（最新 50 条）；
    2. 每页内部按时间升序返回，便于前端直接渲染。
    """

    normalized_uuid = conversation_uuid.strip()
    if not normalized_uuid:
        raise ServiceException(code=ResponseCode.BAD_REQUEST, message="会话UUID不能为空")

    current_user_id = get_user_id()
    conversation_id = _load_admin_conversation(
        conversation_uuid=normalized_uuid,
        user_id=current_user_id,
    )

    skip = (page_request.page_num - 1) * page_request.page_size
    message_documents = list_messages(
        conversation_id=conversation_id,
        limit=page_request.page_size,
        skip=skip,
        ascending=False,
    )

    result: list[ConversationMessageResponse] = []
    for document in reversed(message_documents):
        role = "user" if document.role == MessageRole.USER else "ai"
        payload: dict[str, Any] = {
            "id": document.uuid,
            "role": role,
            "content": document.content,
        }
        if role == "ai":
            payload["status"] = document.status.value
            raw_thinking = getattr(document, "thinking", None)
            if isinstance(raw_thinking, str) and raw_thinking.strip():
                payload["thinking"] = raw_thinking
        result.append(ConversationMessageResponse.model_validate(payload))
    return result


def delete_conversation(
        *,
        conversation_uuid: str,
) -> None:
    """
    删除当前用户的管理助手会话。

    Args:
        conversation_uuid: 会话 UUID。

    Raises:
        ServiceException:
            - BAD_REQUEST: 会话 UUID 为空；
            - NOT_FOUND: 会话不存在或无权限；
            - DATABASE_ERROR: 数据库异常。
    """

    normalized_uuid = conversation_uuid.strip()
    if not normalized_uuid:
        raise ServiceException(code=ResponseCode.BAD_REQUEST, message="会话UUID不能为空")

    current_user_id = get_user_id()
    deleted = delete_admin_conversation(
        conversation_uuid=normalized_uuid,
        user_id=current_user_id,
    )
    if not deleted:
        raise ServiceException(code=ResponseCode.NOT_FOUND, message="会话不存在")


def update_conversation_title(
        *,
        conversation_uuid: str,
        title: str,
) -> str:
    """
    更新当前用户管理助手会话标题。

    Args:
        conversation_uuid: 会话 UUID。
        title: 新标题。

    Returns:
        str: 归一化后的标题（strip 后）。

    Raises:
        ServiceException:
            - BAD_REQUEST: 会话 UUID 或标题为空；
            - NOT_FOUND: 会话不存在或无权限；
            - DATABASE_ERROR: 数据库异常。
    """

    normalized_uuid = conversation_uuid.strip()
    if not normalized_uuid:
        raise ServiceException(code=ResponseCode.BAD_REQUEST, message="会话UUID不能为空")

    normalized_title = title.strip()
    if not normalized_title:
        raise ServiceException(code=ResponseCode.BAD_REQUEST, message="会话标题不能为空")

    current_user_id = get_user_id()
    updated = update_admin_conversation_title(
        conversation_uuid=normalized_uuid,
        user_id=current_user_id,
        title=normalized_title,
    )
    if not updated:
        raise ServiceException(code=ResponseCode.NOT_FOUND, message="会话不存在")

    return normalized_title


def generate_title(question: str) -> str:
    """根据用户输入生成标题。"""

    system_prompt = load_prompt("_system/generate_title.md").strip()
    if not question:
        return "未知标题"

    llm_model = create_chat_model(
        temperature=1.0
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ]
    response = llm_model.invoke(
        messages
    )
    content = getattr(response, "content", "")
    if isinstance(content, str) and content.strip():
        return content.strip()
    return "未知标题"
