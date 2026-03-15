import json
from types import SimpleNamespace

import pytest
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage

from app.core.agent.agent_orchestrator import AssistantStreamConfig
from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException
from app.schemas.base_request import PageRequest
from app.schemas.document.conversation import ConversationDocument, ConversationListItem, ConversationType
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
        lambda *, memory_type, conversation_uuid, user_id: (
            captured.update(
                {
                    "memory_type": memory_type,
                    "memory_conversation_uuid": conversation_uuid,
                    "memory_user_id": user_id,
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
    }


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
    )

    assert isinstance(response, StreamingResponse)
    assert captured["question"] == "代理测试"
    assert captured["callback_kwargs"]["workflow_name"] == service_module.CLIENT_WORKFLOW_NAME
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


def test_conversation_messages_reads_client_conversation_history(monkeypatch):
    monkeypatch.setattr(service_module, "get_user_id", lambda: 66)
    monkeypatch.setattr(
        service_module,
        "_load_client_conversation",
        lambda *, conversation_uuid, user_id: "507f1f77bcf86cd799439099",
    )
    monkeypatch.setattr(service_module, "count_messages", lambda *, conversation_id: 2)
    monkeypatch.setattr(
        service_module,
        "list_messages",
        lambda **_kwargs: [
            SimpleNamespace(
                uuid="ai-1",
                role=MessageRole.AI,
                status=MessageStatus.SUCCESS,
                content="您好",
                thinking="思考文本",
            ),
            SimpleNamespace(
                uuid="user-1",
                role=MessageRole.USER,
                status=MessageStatus.SUCCESS,
                content="你好",
                thinking=None,
            ),
        ],
    )

    rows, total = service_module.conversation_messages(
        conversation_uuid="client-conv-1",
        page_request=PageRequest(page_num=1, page_size=20),
    )

    assert total == 2
    assert [item.model_dump(by_alias=True, exclude_none=True) for item in rows] == [
        {"id": "user-1", "role": "user", "content": "你好"},
        {
            "id": "ai-1",
            "role": "ai",
            "content": "您好",
            "thinking": "思考文本",
            "status": "success",
        },
    ]
