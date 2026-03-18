from __future__ import annotations

import uuid
from typing import Any

from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from app.agent.client.workflow import build_graph
from app.core.agent.agent_orchestrator import AssistantStreamConfig, create_streaming_response
from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException
from app.core.langsmith import build_langsmith_runnable_config
from app.core.security.auth_context import get_user_id
from app.schemas.admin_assistant_history import ConversationMessageResponse
from app.schemas.base_request import PageRequest
from app.schemas.document.conversation import ConversationListItem
from app.schemas.document.message import MessageRole
from app.services.admin_assistant_service import (
    ConversationContext,
    _build_assistant_message_callback,
    _build_conversation_created_event,
    _build_initial_state,
    _build_message_prepared_event,
    _map_exception,
    _persist_user_message,
    _serialize_cards_for_history,
    _schedule_background_task,
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
from app.services.message_service import count_messages, hide_message_card, list_messages

CLIENT_WORKFLOW_NAME = "client_assistant_graph"
CLIENT_WORKFLOW = build_graph()
ClientCardAction = dict[str, str]


def _invoke_client_workflow(state: dict[str, Any]) -> dict[str, Any]:
    """同步执行 client workflow。"""

    config = build_langsmith_runnable_config(
        run_name=CLIENT_WORKFLOW_NAME,
        tags=["client-assistant", "langgraph"],
        metadata={"entrypoint": "api.client_assistant.chat"},
    )
    if config:
        return CLIENT_WORKFLOW.invoke(state, config=config)
    return CLIENT_WORKFLOW.invoke(state)


def _build_stream_config() -> dict | None:
    """构建 client 流式执行 tracing 配置。"""

    return build_langsmith_runnable_config(
        run_name=CLIENT_WORKFLOW_NAME,
        tags=["client-assistant", "langgraph"],
        metadata={"entrypoint": "api.client_assistant.chat"},
    )


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
        card_action: ClientCardAction | None = None,
) -> ConversationContext:
    """准备已存在的 client 会话上下文。"""

    conversation_id = _load_client_conversation(
        conversation_uuid=conversation_uuid,
        user_id=user_id,
    )
    _apply_card_action(
        conversation_id=conversation_id,
        card_action=card_action,
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


def _prepare_conversation_context(
        *,
        question: str,
        user_id: int,
        conversation_uuid: str | None,
        assistant_message_uuid: str,
        card_action: ClientCardAction | None = None,
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
        card_action=card_action,
    )


def _apply_card_action(
        *,
        conversation_id: str,
        card_action: ClientCardAction | None,
) -> None:
    """在加载旧会话上下文前应用前端上报的卡片点击事件。"""

    if card_action is None:
        return
    hide_message_card(
        conversation_id=conversation_id,
        message_uuid=str(card_action.get("message_id") or "").strip(),
        card_uuid=str(card_action.get("card_uuid") or "").strip(),
    )


def assistant_chat(
        *,
        question: str,
        conversation_uuid: str | None = None,
        card_action: ClientCardAction | None = None,
) -> StreamingResponse:
    """客户端助手聊天入口（SSE 流式返回）。"""

    current_user_id = get_user_id()
    assistant_message_uuid = str(uuid.uuid4())
    context = _prepare_conversation_context(
        question=question,
        user_id=current_user_id,
        conversation_uuid=conversation_uuid,
        assistant_message_uuid=assistant_message_uuid,
        card_action=card_action,
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
        workflow=CLIENT_WORKFLOW,
        build_initial_state=lambda q: _build_initial_state(
            q,
            history_messages=context.history_messages,
        ),
        extract_final_content=lambda state: str(state.get("result") or ""),
        should_stream_token=_should_stream_token,
        build_stream_config=_build_stream_config,
        invoke_sync=_invoke_client_workflow,
        map_exception=_map_exception,
        on_answer_completed=_build_assistant_message_callback(
            conversation_id=context.conversation_id,
            assistant_message_uuid=context.assistant_message_uuid,
            workflow_name=CLIENT_WORKFLOW_NAME,
        ),
        initial_emitted_events=context.initial_emitted_events,
    )
    return create_streaming_response(question, stream_config)


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
