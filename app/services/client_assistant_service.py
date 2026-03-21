from __future__ import annotations

import asyncio
import uuid
from typing import Any

from fastapi.responses import StreamingResponse
from langgraph.types import Command
from langchain_core.messages import HumanMessage

from app.agent.client.domain.consultation.agent import has_pending_consultation_interrupt
from app.agent.client.domain.consultation.graph import _CONSULTATION_GRAPH
from app.agent.client.domain.consultation.helpers import (
    build_consultation_followup_card_response,
    resolve_consultation_result_text,
    resolve_interrupt_payload,
)
from app.agent.client.workflow import build_graph
from app.core.agent.agent_orchestrator import AssistantStreamConfig
from app.core.agent.run_event_store import LocalRunHandle
from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException
from app.core.langsmith import build_langsmith_runnable_config
from app.core.security.auth_context import get_user_id
from app.core.speech import build_message_tts_stream
from app.schemas.assistant_run import (
    AssistantRunStatus,
    AssistantRunStopResponse,
    AssistantRunSubmitResponse,
)
from app.schemas.admin_assistant_history import ConversationMessageResponse
from app.schemas.base_request import PageRequest
from app.schemas.document.conversation import ConversationListItem, ConversationType
from app.schemas.document.message import MessageRole
from app.services.admin_assistant_service import (
    ConversationContext,
    RUN_EVENT_STORE,
    _build_assistant_message_callback,
    _build_attach_streaming_response,
    _build_background_run_done_callback,
    _build_conversation_created_event,
    _build_message_prepared_event,
    _create_placeholder_assistant_message,
    _map_exception,
    _persist_user_message,
    _run_assistant_workflow_in_background,
    _serialize_cards_for_history,
    _schedule_title_generation,
    _should_stream_token,
)
from app.services.conversation_service import (
    add_client_conversation,
    delete_client_conversation,
    get_client_conversation,
    list_client_conversations,
    update_client_conversation_title,
)
from app.services.memory_service import load_memory, resolve_assistant_memory_mode
from app.services.message_service import count_messages, hide_visible_cards_in_conversation, list_messages

CLIENT_WORKFLOW_NAME = "client_assistant_graph"
CLIENT_WORKFLOW = build_graph()

CONSULTATION_WORKFLOW_NAME = "client_consultation_graph"
"""consultation resume 直连子图时使用的内部 workflow 名称。"""


def _merge_runnable_config(
        *,
        run_name: str,
        conversation_uuid: str,
) -> dict[str, Any]:
    """
    功能描述：
        构造并合并 LangSmith tracing 与 `thread_id`。

    参数说明：
        run_name (str): workflow 运行名称。
        conversation_uuid (str): 会话 UUID，同时作为 LangGraph `thread_id`。

    返回值：
        dict[str, Any]: 最终 runnable config。

    异常说明：
        无。
    """

    base_config = build_langsmith_runnable_config(
        run_name=run_name,
        tags=["client-assistant", "langgraph"],
        metadata={"entrypoint": "api.client_assistant.chat"},
    ) or {}
    configurable = dict(base_config.get("configurable") or {})
    configurable["thread_id"] = conversation_uuid
    merged_config = dict(base_config)
    merged_config["configurable"] = configurable
    return merged_config


def _invoke_workflow_with_config(
        *,
        workflow: Any,
        workflow_input: Any,
        runnable_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    功能描述：
        同步执行 workflow，并在需要时透传 runnable config。

    参数说明：
        workflow (Any): 已编译的 workflow。
        workflow_input (Any): workflow 输入，可以是 state 或 `Command`。
        runnable_config (dict[str, Any] | None): runnable config。

    返回值：
        dict[str, Any]: workflow 最终返回状态。

    异常说明：
        无；执行异常由调用方感知。
    """

    if runnable_config:
        return workflow.invoke(workflow_input, config=runnable_config)
    return workflow.invoke(workflow_input)


def _build_client_initial_state(
        *,
        history_messages: list[Any],
) -> dict[str, Any]:
    """
    功能描述：
        构造 client 主图的初始状态。

    参数说明：
        history_messages (list[Any]): 当前会话历史消息列表。

    返回值：
        dict[str, Any]: client workflow 初始状态。

    异常说明：
        无。
    """

    base_history = list(history_messages)
    return {
        "routing": {
            "route_targets": [],
            "task_difficulty": "normal",
        },
        "context": "",
        "history_messages": base_history,
        "execution_traces": [],
        "token_usage": None,
        "result": "",
        "messages": list(base_history),
    }


def _build_consultation_interrupt_responses(state: dict[str, Any]) -> list[Any]:
    """
    功能描述：
        将 consultation 的 interrupt state 转换为即时 SSE 响应。

    参数说明：
        state (dict[str, Any]): workflow 最新状态。

    返回值：
        list[Any]: 需要立即发送给前端的响应列表。

    异常说明：
        无；不命中 consultation interrupt 时返回空列表。
    """

    interrupt_payload = resolve_interrupt_payload(state)
    if interrupt_payload is None:
        return []
    reply_text = str(
        interrupt_payload.get("reply_text")
        or interrupt_payload.get("question_text")
        or ""
    ).strip()
    question_text = str(interrupt_payload.get("question_text") or "").strip()
    options = list(interrupt_payload.get("options") or [])
    if not reply_text or not question_text:
        return []
    return [
        build_consultation_followup_card_response(
            title=question_text,
            description=reply_text,
            options=options,
        )
    ]


def _load_client_conversation(
        *,
        conversation_uuid: str,
        user_id: int,
) -> str:
    """加载 client 会话并返回 Mongo 会话 ID。"""

    conversation = get_client_conversation(
        conversation_uuid=conversation_uuid,
        user_id=user_id,
    )
    if conversation is None:
        raise ServiceException(code=ResponseCode.NOT_FOUND, message="会话不存在")

    conversation_id = conversation.id
    if conversation_id is None:
        raise ServiceException(code=ResponseCode.DATABASE_ERROR, message="会话数据异常")
    return str(conversation_id)


def _prepare_new_conversation(
        *,
        question: str,
        user_id: int,
        assistant_message_uuid: str,
) -> ConversationContext:
    """准备新的 client 会话上下文。"""

    conversation_uuid = str(uuid.uuid4())
    conversation_id = add_client_conversation(
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
    """准备已存在的 client 会话上下文。"""

    conversation_id = _load_client_conversation(
        conversation_uuid=conversation_uuid,
        user_id=user_id,
    )
    _hide_visible_conversation_cards(
        conversation_id=conversation_id,
    )
    memory = load_memory(
        memory_type=resolve_assistant_memory_mode(),
        conversation_uuid=conversation_uuid,
        user_id=user_id,
        include_history_hidden=False,
    )
    history_messages = [*list(memory.messages), HumanMessage(content=question)]
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


def _prepare_existing_resume_conversation(
        *,
        conversation_uuid: str,
        user_id: int,
        assistant_message_uuid: str,
) -> ConversationContext:
    """
    功能描述：
        为 consultation resume 场景准备会话上下文。

    参数说明：
        conversation_uuid (str): 会话 UUID。
        user_id (int): 当前用户 ID。
        assistant_message_uuid (str): 本轮 AI 消息 UUID。

    返回值：
        ConversationContext: resume 专用会话上下文。

    异常说明：
        ServiceException: 会话不存在或数据库异常时抛出。
    """

    conversation_id = _load_client_conversation(
        conversation_uuid=conversation_uuid,
        user_id=user_id,
    )
    _hide_visible_conversation_cards(
        conversation_id=conversation_id,
    )
    return ConversationContext(
        conversation_uuid=conversation_uuid,
        conversation_id=conversation_id,
        assistant_message_uuid=assistant_message_uuid,
        history_messages=[],
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
    """统一准备 client 会话上下文。"""

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


def _hide_visible_conversation_cards(
        *,
        conversation_id: str,
) -> None:
    """在加载旧会话 memory 前批量隐藏当前仍可见的 AI 卡片。"""

    hide_visible_cards_in_conversation(
        conversation_id=conversation_id,
    )


def assistant_chat(
        *,
        question: str,
        conversation_uuid: str | None = None,
) -> AssistantRunSubmitResponse:
    """客户端助手聊天提交入口（创建后台 run 并返回运行态）。"""

    current_user_id = get_user_id()
    assistant_message_uuid = str(uuid.uuid4())
    should_resume_consultation = (
            conversation_uuid is not None
            and has_pending_consultation_interrupt(conversation_uuid=conversation_uuid)
    )
    if should_resume_consultation:
        context = _prepare_existing_resume_conversation(
            conversation_uuid=str(conversation_uuid),
            user_id=current_user_id,
            assistant_message_uuid=assistant_message_uuid,
        )
    else:
        context = _prepare_conversation_context(
            question=question,
            user_id=current_user_id,
            conversation_uuid=conversation_uuid,
            assistant_message_uuid=assistant_message_uuid,
        )

    created_meta = RUN_EVENT_STORE.create_run(
        conversation_uuid=context.conversation_uuid,
        user_id=current_user_id,
        conversation_type=ConversationType.CLIENT.value,
        assistant_message_uuid=context.assistant_message_uuid,
    )
    if created_meta is None:
        active_meta = RUN_EVENT_STORE.get_run_meta(
            conversation_uuid=context.conversation_uuid,
        )
        if active_meta is not None and active_meta.status == AssistantRunStatus.RUNNING:
            raise ServiceException(
                code=ResponseCode.CONFLICT,
                message="当前会话已有正在输出的回答",
                data=AssistantRunSubmitResponse(
                    conversation_uuid=context.conversation_uuid,
                    message_uuid=active_meta.assistant_message_uuid,
                    run_status=active_meta.status,
                ).model_dump(),
            )
        raise ServiceException(
            code=ResponseCode.SERVICE_UNAVAILABLE,
            message="创建助手运行态失败",
        )

    _persist_user_message(
        conversation_id=context.conversation_id,
        question=question,
    )
    _create_placeholder_assistant_message(
        conversation_id=context.conversation_id,
        message_uuid=context.assistant_message_uuid,
    )

    cancel_event = asyncio.Event()
    resolved_workflow = _CONSULTATION_GRAPH if should_resume_consultation else CLIENT_WORKFLOW
    resolved_run_name = (
        CONSULTATION_WORKFLOW_NAME
        if should_resume_consultation
        else CLIENT_WORKFLOW_NAME
    )
    runnable_config = _merge_runnable_config(
        run_name=resolved_run_name,
        conversation_uuid=context.conversation_uuid,
    )
    stream_config = AssistantStreamConfig(
        workflow=resolved_workflow,
        build_initial_state=(
            (lambda _question: Command(resume=question))
            if should_resume_consultation
            else (
                lambda _question: _build_client_initial_state(
                    history_messages=context.history_messages,
                )
            )
        ),
        extract_final_content=(
            (lambda state: resolve_consultation_result_text(state))
            if should_resume_consultation
            else (lambda state: str(state.get("result") or ""))
        ),
        should_stream_token=(
            (lambda _node, _state: False)
            if should_resume_consultation
            else _should_stream_token
        ),
        build_stream_config=lambda: runnable_config,
        invoke_sync=lambda workflow_input: _invoke_workflow_with_config(
            workflow=resolved_workflow,
            workflow_input=workflow_input,
            runnable_config=runnable_config,
        ),
        map_exception=_map_exception,
        on_answer_completed=_build_assistant_message_callback(
            conversation_id=context.conversation_id,
            assistant_message_uuid=context.assistant_message_uuid,
            workflow_name=resolved_run_name,
        ),
        initial_emitted_events=context.initial_emitted_events,
        is_cancel_requested=lambda: (
                cancel_event.is_set()
                or RUN_EVENT_STORE.is_cancel_requested(
            conversation_uuid=context.conversation_uuid,
        )
        ),
        build_interrupt_responses=_build_consultation_interrupt_responses,
    )
    background_task = asyncio.create_task(
        _run_assistant_workflow_in_background(
            question=question,
            context=context,
            stream_config=stream_config,
        )
    )
    background_task.add_done_callback(
        _build_background_run_done_callback(
            conversation_uuid=context.conversation_uuid,
        )
    )
    RUN_EVENT_STORE.register_local_handle(
        conversation_uuid=context.conversation_uuid,
        handle=LocalRunHandle(
            task=background_task,
            cancel_event=cancel_event,
        ),
    )
    return AssistantRunSubmitResponse(
        conversation_uuid=context.conversation_uuid,
        message_uuid=context.assistant_message_uuid,
        run_status=AssistantRunStatus.RUNNING,
    )


def assistant_chat_stream(
        *,
        conversation_uuid: str,
        last_event_id: str | None = None,
) -> StreamingResponse:
    """attach 到客户端助手当前的流式 run。"""

    current_user_id = get_user_id()
    _load_client_conversation(
        conversation_uuid=conversation_uuid,
        user_id=current_user_id,
    )
    meta = RUN_EVENT_STORE.get_run_meta(conversation_uuid=conversation_uuid)
    if meta is None:
        raise ServiceException(
            code=ResponseCode.NOT_FOUND,
            message="当前会话没有可连接的流式输出",
        )
    return _build_attach_streaming_response(
        conversation_uuid=conversation_uuid,
        last_event_id=last_event_id,
    )


def assistant_chat_stop(
        *,
        conversation_uuid: str,
) -> AssistantRunStopResponse:
    """停止客户端助手当前会话的流式 run。"""

    current_user_id = get_user_id()
    _load_client_conversation(
        conversation_uuid=conversation_uuid,
        user_id=current_user_id,
    )
    meta = RUN_EVENT_STORE.request_cancel(conversation_uuid=conversation_uuid)
    if meta is None or meta.status != AssistantRunStatus.RUNNING:
        raise ServiceException(
            code=ResponseCode.NOT_FOUND,
            message="当前会话没有运行中的输出",
        )
    return AssistantRunStopResponse(
        conversation_uuid=conversation_uuid,
        message_uuid=meta.assistant_message_uuid,
        run_status=meta.status,
        stop_requested=True,
    )


def assistant_message_tts_stream(
        *,
        message_uuid: str,
) -> StreamingResponse:
    """
    客户端助手消息转语音（HTTP chunked audio stream）。

    说明：
    - 先基于 `message_uuid` 校验消息存在性、client 会话归属与消息角色；
    - 校验通过后建立上游 Volcengine 双向 TTS websocket；
    - 下游以音频字节流（chunked）持续返回给前端。

    Args:
        message_uuid: 目标 AI 消息 UUID。

    Returns:
        StreamingResponse: 下游音频流响应对象。
    """

    current_user_id = get_user_id()
    tts_stream = build_message_tts_stream(
        message_uuid=message_uuid,
        user_id=current_user_id,
        conversation_type=ConversationType.CLIENT,
    )
    return StreamingResponse(
        tts_stream.audio_stream,
        media_type=tts_stream.media_type,
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def conversation_list(
        *,
        page_request: PageRequest,
) -> tuple[list[ConversationListItem], int]:
    """分页查询当前用户的 client 会话列表。"""

    current_user_id = get_user_id()
    return list_client_conversations(
        user_id=current_user_id,
        page_num=page_request.page_num,
        page_size=page_request.page_size,
    )


def conversation_messages(
        *,
        conversation_uuid: str,
        page_request: PageRequest,
) -> tuple[list[ConversationMessageResponse], int]:
    """分页查询当前用户某个 client 会话的历史消息。"""

    normalized_uuid = conversation_uuid.strip()
    if not normalized_uuid:
        raise ServiceException(code=ResponseCode.BAD_REQUEST, message="会话UUID不能为空")

    current_user_id = get_user_id()
    conversation_id = _load_client_conversation(
        conversation_uuid=normalized_uuid,
        user_id=current_user_id,
    )

    skip = (page_request.page_num - 1) * page_request.page_size
    total = count_messages(
        conversation_id=conversation_id,
        history_hidden=False,
    )
    message_documents = list_messages(
        conversation_id=conversation_id,
        limit=page_request.page_size,
        skip=skip,
        ascending=False,
        history_hidden=False,
    )

    result: list[ConversationMessageResponse] = []
    for document in reversed(message_documents):
        role = "user" if document.role == MessageRole.USER else "ai"
        raw_thinking = getattr(document, "thinking", None)
        normalized_thinking = (
            raw_thinking.strip()
            if isinstance(raw_thinking, str) and raw_thinking.strip()
            else None
        )
        serialized_cards = None
        if role == "ai":
            serialized_cards = _serialize_cards_for_history(
                getattr(document, "cards", None),
                hidden_card_uuids=getattr(document, "hidden_card_uuids", None),
            )
            if not document.content.strip() and normalized_thinking is None and serialized_cards is None:
                continue
        payload: dict[str, Any] = {
            "id": document.uuid,
            "role": role,
            "content": document.content,
        }
        if role == "ai":
            payload["status"] = document.status.value
            if normalized_thinking is not None:
                payload["thinking"] = normalized_thinking
            if serialized_cards is not None:
                payload["cards"] = serialized_cards
        result.append(ConversationMessageResponse.model_validate(payload))
    return result, total


def delete_conversation(
        *,
        conversation_uuid: str,
) -> None:
    """
    删除当前用户的客户端助手会话。

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
    deleted = delete_client_conversation(
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
    更新当前用户客户端助手会话标题。

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
    updated = update_client_conversation_title(
        conversation_uuid=normalized_uuid,
        user_id=current_user_id,
        title=normalized_title,
    )
    if not updated:
        raise ServiceException(code=ResponseCode.NOT_FOUND, message="会话不存在")

    return normalized_title
