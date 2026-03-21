import json
from types import SimpleNamespace

import pytest
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage

from app.core.agent.agent_orchestrator import AssistantStreamConfig
from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException
from app.core.speech.tts.client import MessageTtsStream
from app.schemas.base_request import PageRequest
from app.schemas.document.conversation import ConversationListItem, ConversationType
from app.schemas.document.message import MessageRole, MessageStatus
from app.services import client_assistant_service as service_module


def test_prepare_new_conversation_uses_client_conversation_storage(monkeypatch):
    scheduled_title_calls: list[dict] = []

    monkeypatch.setattr(service_module.uuid, "uuid4", lambda: "client-conv-uuid")
    monkeypatch.setattr(
        service_module,
        "add_client_conversation",
        lambda *, conversation_uuid, user_id: f"db-{conversation_uuid}-{user_id}",
    )
    monkeypatch.setattr(
        service_module,
        "_schedule_title_generation",
        lambda **kwargs: scheduled_title_calls.append(kwargs),
    )

    context = service_module._prepare_new_conversation(
        question="我要咨询",
        user_id=100,
        assistant_message_uuid="assistant-msg-1",
    )

    assert isinstance(context, service_module.ConversationContext)
    assert context.conversation_uuid == "client-conv-uuid"
    assert context.conversation_id == "db-client-conv-uuid-100"
    assert [message.content for message in context.history_messages] == ["我要咨询"]
    assert context.initial_emitted_events[0].meta == {
        "conversation_uuid": "client-conv-uuid",
        "message_uuid": "assistant-msg-1",
    }
    assert scheduled_title_calls == [
        {"conversation_uuid": "client-conv-uuid", "question": "我要咨询"}
    ]


def test_prepare_existing_conversation_loads_client_memory(monkeypatch):
    captured: dict = {}
    expected_history = [
        HumanMessage(content="历史问题"),
        AIMessage(content="历史回答"),
    ]

    monkeypatch.setattr(
        service_module,
        "_load_client_conversation",
        lambda *, conversation_uuid, user_id: (
            captured.update(
                {
                    "conversation_uuid": conversation_uuid,
                    "user_id": user_id,
                }
            ),
            "507f1f77bcf86cd799439011",
        )[-1],
    )
    monkeypatch.setattr(service_module, "resolve_assistant_memory_mode", lambda: "summary")
    monkeypatch.setattr(
        service_module,
        "load_memory",
        lambda *, memory_type, conversation_uuid, user_id, include_history_hidden: (
            captured.update(
                {
                    "memory_type": memory_type,
                    "memory_conversation_uuid": conversation_uuid,
                    "memory_user_id": user_id,
                    "include_history_hidden": include_history_hidden,
                }
            ),
            SimpleNamespace(messages=expected_history),
        )[-1],
    )

    context = service_module._prepare_existing_conversation(
        conversation_uuid="client-conv-1",
        user_id=100,
        question="本轮问题",
        assistant_message_uuid="assistant-msg-2",
    )

    assert [message.content for message in context.history_messages] == [
        "历史问题",
        "历史回答",
        "本轮问题",
    ]
    assert context.initial_emitted_events[0].meta == {"message_uuid": "assistant-msg-2"}
    assert captured == {
        "conversation_uuid": "client-conv-1",
        "user_id": 100,
        "memory_type": "summary",
        "memory_conversation_uuid": "client-conv-1",
        "memory_user_id": 100,
        "include_history_hidden": False,
    }


def test_prepare_existing_conversation_applies_card_action_before_loading_memory(monkeypatch):
    calls: list[tuple[str, dict]] = []

    monkeypatch.setattr(
        service_module,
        "_load_client_conversation",
        lambda *, conversation_uuid, user_id: "507f1f77bcf86cd799439011",
    )
    monkeypatch.setattr(
        service_module,
        "_apply_card_action",
        lambda *, conversation_id, card_action: calls.append(
            (
                "card_action",
                {
                    "conversation_id": conversation_id,
                    "card_action": card_action,
                },
            )
        ),
    )
    monkeypatch.setattr(service_module, "resolve_assistant_memory_mode", lambda: "window")
    monkeypatch.setattr(
        service_module,
        "load_memory",
        lambda *, memory_type, conversation_uuid, user_id, include_history_hidden: (
            calls.append(
                (
                    "load_memory",
                    {
                        "memory_type": memory_type,
                        "conversation_uuid": conversation_uuid,
                        "user_id": user_id,
                        "include_history_hidden": include_history_hidden,
                    },
                )
            ),
            SimpleNamespace(messages=[]),
        )[-1],
    )

    service_module._prepare_existing_conversation(
        conversation_uuid="client-conv-1",
        user_id=100,
        question="本轮问题",
        assistant_message_uuid="assistant-msg-2",
        card_action={
            "type": "click",
            "message_id": "msg-1",
            "card_uuid": "card-1",
        },
    )

    assert calls == [
        (
            "card_action",
            {
                "conversation_id": "507f1f77bcf86cd799439011",
                "card_action": {
                    "type": "click",
                    "message_id": "msg-1",
                    "card_uuid": "card-1",
                },
            },
        ),
        (
            "load_memory",
            {
                "memory_type": "window",
                "conversation_uuid": "client-conv-1",
                "user_id": 100,
                "include_history_hidden": False,
            },
        ),
    ]


def test_load_client_conversation_raises_not_found_when_missing(monkeypatch):
    monkeypatch.setattr(
        service_module,
        "get_client_conversation",
        lambda *, conversation_uuid, user_id: None,
    )

    with pytest.raises(ServiceException) as exc_info:
        service_module._load_client_conversation(
            conversation_uuid="missing",
            user_id=100,
        )

    assert exc_info.value.code == ResponseCode.NOT_FOUND.code


def test_assistant_chat_builds_stream_config_with_client_workflow_name(monkeypatch):
    captured: dict = {}
    background_calls: list[dict] = []

    monkeypatch.setattr(service_module, "get_user_id", lambda: 101)
    monkeypatch.setattr(
        service_module.uuid,
        "uuid4",
        lambda: "client-msg-uuid",
    )
    monkeypatch.setattr(
        service_module,
        "_prepare_conversation_context",
        lambda **kwargs: (
            captured.update({"context_kwargs": kwargs}),
            service_module.ConversationContext(
                conversation_uuid="client-conv-1",
                conversation_id="507f1f77bcf86cd799439011",
                assistant_message_uuid="client-msg-uuid",
                history_messages=[HumanMessage(content="代理测试")],
                initial_emitted_events=(),
                is_new_conversation=True,
            ),
        )[-1],
    )
    monkeypatch.setattr(
        service_module,
        "_schedule_background_task",
        lambda *, task_name, func, kwargs: background_calls.append(
            {"task_name": task_name, "func": func, "kwargs": kwargs}
        ),
    )
    monkeypatch.setattr(
        service_module,
        "_build_assistant_message_callback",
        lambda **kwargs: captured.update({"callback_kwargs": kwargs}) or (lambda *_args, **_kw: None),
    )

    def _fake_create_streaming_response(question: str, config: AssistantStreamConfig):
        captured["question"] = question
        captured["config"] = config

        async def _stream():
            yield (
                    "data: "
                    + json.dumps(
                {
                    "content": {"text": ""},
                    "type": "answer",
                    "is_end": True,
                    "timestamp": 1,
                },
                ensure_ascii=False,
            )
                    + "\n\n"
            )

        return StreamingResponse(_stream(), media_type="text/event-stream")

    monkeypatch.setattr(
        service_module,
        "create_streaming_response",
        _fake_create_streaming_response,
    )

    response = service_module.assistant_chat(
        question="代理测试",
        conversation_uuid=None,
        card_action={
            "type": "click",
            "message_id": "msg-1",
            "card_uuid": "card-1",
        },
    )

    assert isinstance(response, StreamingResponse)
    assert captured["question"] == "代理测试"
    assert captured["context_kwargs"]["card_action"] == {
        "type": "click",
        "message_id": "msg-1",
        "card_uuid": "card-1",
    }
    assert captured["callback_kwargs"]["workflow_name"] == service_module.CLIENT_WORKFLOW_NAME
    assert "persist_cards" not in captured["callback_kwargs"]
    assert background_calls == [
        {
            "task_name": "persist_user_message",
            "func": service_module._persist_user_message,
            "kwargs": {
                "conversation_id": "507f1f77bcf86cd799439011",
                "question": "代理测试",
            },
        }
    ]


def test_conversation_list_uses_client_storage(monkeypatch):
    captured: dict = {}

    monkeypatch.setattr(service_module, "get_user_id", lambda: 55)
    monkeypatch.setattr(
        service_module,
        "list_client_conversations",
        lambda *, user_id, page_num, page_size: (
            captured.update(
                {
                    "user_id": user_id,
                    "page_num": page_num,
                    "page_size": page_size,
                }
            ),
            ([ConversationListItem(conversation_uuid="client-conv-1", title="标题1")], 1),
        )[-1],
    )

    rows, total = service_module.conversation_list(
        page_request=PageRequest(page_num=2, page_size=10),
    )

    assert captured == {"user_id": 55, "page_num": 2, "page_size": 10}
    assert rows[0].conversation_uuid == "client-conv-1"
    assert total == 1


def test_assistant_message_tts_stream_returns_chunked_audio_with_expected_headers(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(service_module, "get_user_id", lambda: 77)

    async def _audio_stream():
        yield b"chunk-a"
        yield b"chunk-b"

    monkeypatch.setattr(
        service_module,
        "build_message_tts_stream",
        lambda **kwargs: (
            captured.update(kwargs),
            MessageTtsStream(audio_stream=_audio_stream(), media_type="audio/mpeg"),
        )[-1],
    )

    response = service_module.assistant_message_tts_stream(
        message_uuid="client-msg-1",
    )

    assert isinstance(response, StreamingResponse)
    assert response.media_type == "audio/mpeg"
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert captured == {
        "message_uuid": "client-msg-1",
        "user_id": 77,
        "conversation_type": ConversationType.CLIENT,
    }


def test_conversation_messages_reads_client_conversation_history(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(service_module, "get_user_id", lambda: 66)
    monkeypatch.setattr(
        service_module,
        "_load_client_conversation",
        lambda *, conversation_uuid, user_id: "507f1f77bcf86cd799439099",
    )
    monkeypatch.setattr(
        service_module,
        "count_messages",
        lambda *, conversation_id, history_hidden=None: (
            captured.update(
                {
                    "count_conversation_id": conversation_id,
                    "count_history_hidden": history_hidden,
                }
            ),
            2,
        )[-1],
    )
    monkeypatch.setattr(
        service_module,
        "list_messages",
        lambda **kwargs: (
            captured.update(
                {
                    "list_conversation_id": kwargs["conversation_id"],
                    "list_history_hidden": kwargs.get("history_hidden"),
                }
            ),
            [
            SimpleNamespace(
                uuid="ai-1",
                role=MessageRole.AI,
                status=MessageStatus.SUCCESS,
                content="",
                thinking="思考文本",
                hidden_card_uuids=["card-2"],
                cards=[
                    {
                        "id": "card-1",
                        "type": "product-card",
                        "data": {
                            "title": "为您推荐以下商品",
                            "products": [{"id": "1001", "name": "商品1001"}],
                        },
                    },
                    {
                        "id": "card-2",
                        "type": "product-purchase-card",
                        "data": {
                            "title": "请确认要购买的商品",
                            "products": [
                                {
                                    "id": "1002",
                                    "name": "商品1002",
                                    "price": "19.90",
                                    "quantity": 2,
                                }
                            ],
                            "total_price": "39.80",
                        },
                    }
                ],
            ),
            SimpleNamespace(
                uuid="user-1",
                role=MessageRole.USER,
                status=MessageStatus.SUCCESS,
                content="你好",
                thinking=None,
            ),
        ],
        )[-1],
    )

    rows, total = service_module.conversation_messages(
        conversation_uuid="client-conv-1",
        page_request=PageRequest(page_num=1, page_size=20),
    )

    assert total == 2
    assert captured == {
        "count_conversation_id": "507f1f77bcf86cd799439099",
        "count_history_hidden": False,
        "list_conversation_id": "507f1f77bcf86cd799439099",
        "list_history_hidden": False,
    }
    assert [item.model_dump(by_alias=True, exclude_none=True) for item in rows] == [
        {"id": "user-1", "role": "user", "content": "你好"},
        {
            "id": "ai-1",
            "role": "ai",
            "content": "",
            "thinking": "思考文本",
            "status": "success",
            "cards": [
                {
                    "card_uuid": "card-1",
                    "type": "product-card",
                    "data": {
                        "title": "为您推荐以下商品",
                        "products": [{"id": "1001", "name": "商品1001"}],
                    },
                }
            ],
        },
    ]


def test_apply_card_action_delegates_to_message_service(monkeypatch):
    captured: dict = {}

    monkeypatch.setattr(
        service_module,
        "hide_message_card",
        lambda *, conversation_id, message_uuid, card_uuid: captured.update(
            {
                "conversation_id": conversation_id,
                "message_uuid": message_uuid,
                "card_uuid": card_uuid,
            }
        ),
    )

    service_module._apply_card_action(
        conversation_id="507f1f77bcf86cd799439011",
        card_action={
            "type": "click",
            "message_id": "msg-1",
            "card_uuid": "card-1",
        },
    )

    assert captured == {
        "conversation_id": "507f1f77bcf86cd799439011",
        "message_uuid": "msg-1",
        "card_uuid": "card-1",
    }


def test_conversation_messages_skips_ai_message_when_all_cards_hidden(monkeypatch):
    monkeypatch.setattr(service_module, "get_user_id", lambda: 66)
    monkeypatch.setattr(
        service_module,
        "_load_client_conversation",
        lambda *, conversation_uuid, user_id: "507f1f77bcf86cd799439099",
    )
    monkeypatch.setattr(
        service_module,
        "count_messages",
        lambda *, conversation_id, history_hidden=None: 0,
    )
    monkeypatch.setattr(
        service_module,
        "list_messages",
        lambda **_kwargs: [
            SimpleNamespace(
                uuid="ai-1",
                role=MessageRole.AI,
                status=MessageStatus.SUCCESS,
                content="",
                thinking=None,
                hidden_card_uuids=["card-1"],
                cards=[
                    {
                        "id": "card-1",
                        "type": "selection-card",
                        "data": {
                            "title": "请选择",
                            "options": ["A", "B"],
                        },
                    }
                ],
            )
        ],
    )

    rows, total = service_module.conversation_messages(
        conversation_uuid="client-conv-1",
        page_request=PageRequest(page_num=1, page_size=20),
    )

    assert total == 0
    assert rows == []


def test_delete_conversation_calls_repository_with_current_user(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(service_module, "get_user_id", lambda: 101)
    monkeypatch.setattr(
        service_module,
        "delete_client_conversation",
        lambda *, conversation_uuid, user_id: (
            captured.update({"conversation_uuid": conversation_uuid, "user_id": user_id}),
            True,
        )[-1],
    )

    service_module.delete_conversation(conversation_uuid="client-conv-1")

    assert captured == {"conversation_uuid": "client-conv-1", "user_id": 101}


def test_delete_conversation_raises_not_found_when_missing(monkeypatch):
    monkeypatch.setattr(service_module, "get_user_id", lambda: 101)
    monkeypatch.setattr(
        service_module,
        "delete_client_conversation",
        lambda **_kwargs: False,
    )

    with pytest.raises(ServiceException) as exc_info:
        service_module.delete_conversation(conversation_uuid="missing-client-conv")
    assert exc_info.value.code == ResponseCode.NOT_FOUND.code


def test_update_conversation_title_returns_normalized_title(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(service_module, "get_user_id", lambda: 101)
    monkeypatch.setattr(
        service_module,
        "update_client_conversation_title",
        lambda *, conversation_uuid, user_id, title: (
            captured.update(
                {
                    "conversation_uuid": conversation_uuid,
                    "user_id": user_id,
                    "title": title,
                }
            ),
            True,
        )[-1],
    )

    title = service_module.update_conversation_title(
        conversation_uuid="client-conv-1",
        title="  新标题  ",
    )

    assert title == "新标题"
    assert captured == {
        "conversation_uuid": "client-conv-1",
        "user_id": 101,
        "title": "新标题",
    }


def test_update_conversation_title_rejects_blank_title(monkeypatch):
    with pytest.raises(ServiceException) as exc_info:
        service_module.update_conversation_title(
            conversation_uuid="client-conv-1",
            title="   ",
        )
    assert exc_info.value.code == ResponseCode.BAD_REQUEST.code


def test_update_conversation_title_raises_not_found_when_missing(monkeypatch):
    monkeypatch.setattr(service_module, "get_user_id", lambda: 101)
    monkeypatch.setattr(
        service_module,
        "update_client_conversation_title",
        lambda **_kwargs: False,
    )

    with pytest.raises(ServiceException) as exc_info:
        service_module.update_conversation_title(
            conversation_uuid="missing-client-conv",
            title="新标题",
        )
    assert exc_info.value.code == ResponseCode.NOT_FOUND.code
