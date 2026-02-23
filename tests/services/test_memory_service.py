from types import SimpleNamespace

import pytest

from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException
from app.schemas.document.message import MessageRole
from app.schemas.memory import Memory
from app.services import memory_service as service_module


def test_load_memory_by_window_returns_ordered_memory(monkeypatch):
    captured: dict = {}

    monkeypatch.setattr(
        service_module,
        "get_conversation",
        lambda *, conversation_uuid, user_id: (
            captured.update(
                {
                    "conversation_uuid": conversation_uuid,
                    "user_id": user_id,
                }
            ),
            SimpleNamespace(id="conv-object-id"),
        )[-1],
    )
    monkeypatch.setattr(
        service_module,
        "list_messages",
        lambda *, conversation_id, limit, ascending: (
            captured.update(
                {
                    "conversation_id": conversation_id,
                    "limit": limit,
                    "ascending": ascending,
                }
            ),
            [
                SimpleNamespace(role=MessageRole.AI, content="A2"),
                SimpleNamespace(role=MessageRole.USER, content="Q1"),
            ],
        )[-1],
    )

    result = service_module.load_memory_by_window(
        conversation_uuid="conv-1",
        user_id=100,
        limit=2,
    )

    assert isinstance(result, Memory)
    assert captured == {
        "conversation_uuid": "conv-1",
        "user_id": 100,
        "conversation_id": "conv-object-id",
        "limit": 2,
        "ascending": False,
    }
    assert [message.type for message in result.messages] == ["human", "ai"]
    assert [message.content for message in result.messages] == ["Q1", "A2"]
    assert all(message.type != "system" for message in result.messages)


def test_load_memory_by_window_raises_not_found_when_conversation_missing(monkeypatch):
    monkeypatch.setattr(service_module, "get_conversation", lambda **_kwargs: None)

    with pytest.raises(ServiceException) as exc_info:
        service_module.load_memory_by_window(
            conversation_uuid="conv-missing",
            user_id=100,
            limit=10,
        )

    assert exc_info.value.code == ResponseCode.NOT_FOUND


def test_load_memory_by_window_raises_database_error_when_conversation_id_missing(monkeypatch):
    monkeypatch.setattr(
        service_module,
        "get_conversation",
        lambda **_kwargs: SimpleNamespace(id=None),
    )

    with pytest.raises(ServiceException) as exc_info:
        service_module.load_memory_by_window(
            conversation_uuid="conv-1",
            user_id=100,
            limit=10,
        )

    assert exc_info.value.code == ResponseCode.DATABASE_ERROR


def test_load_memory_window_dispatches_to_window_loader(monkeypatch):
    monkeypatch.setattr(
        service_module,
        "load_memory_by_window",
        lambda *, conversation_uuid, user_id, limit: Memory(messages=[]),
    )

    result = service_module.load_memory(
        memory_type="window",
        conversation_uuid="conv-1",
        user_id=100,
        limit=20,
    )

    assert isinstance(result, Memory)


def test_load_memory_summary_not_implemented():
    with pytest.raises(NotImplementedError):
        service_module.load_memory_by_summary(
            conversation_uuid="conv-1",
            user_id=100,
        )
