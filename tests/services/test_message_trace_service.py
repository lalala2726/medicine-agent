import datetime

import pytest
from bson import ObjectId

from app.core.codes import ResponseCode
from app.exception.exceptions import ServiceException
from app.schemas.document.message_trace import MessageTraceDocument
from app.services import message_trace_service as service_module


class _DummyInsertResult:
    def __init__(self, inserted_id: str):
        self.inserted_id = inserted_id


class _DummyCollection:
    def __init__(self):
        self.last_inserted: dict | None = None
        self.last_find_one_query: dict | None = None
        self.find_one_result: dict | None = None

    def insert_one(self, document: dict) -> _DummyInsertResult:
        self.last_inserted = document
        return _DummyInsertResult("507f1f77bcf86cd799439021")

    def find_one(self, query: dict) -> dict | None:
        self.last_find_one_query = query
        return self.find_one_result


def test_add_message_trace_inserts_expected_document(monkeypatch):
    """验证 add_message_trace：execution_trace + token 明细可落库。"""

    collection = _DummyCollection()
    monkeypatch.setattr(
        service_module,
        "get_mongo_database",
        lambda: {"message_traces": collection},
    )
    monkeypatch.setattr(service_module, "_resolve_collection_name", lambda: "message_traces")

    result = service_module.add_message_trace(
        message_uuid="msg-1",
        conversation_id="507f1f77bcf86cd799439011",
        execution_trace=[
            {
                "node_name": "chat_agent",
                "model_name": "qwen-max",
                "input_messages": [{"role": "human", "content": "用户问题"}],
                "output_text": "AI回复",
                "tool_calls": [],
            }
        ],
        token_usage_detail={
            "is_complete": True,
            "node_breakdown": [
                {
                    "node_name": "chat_agent",
                    "model_name": "qwen-max",
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "total_tokens": 3,
                    "tool_tokens_total": 0,
                    "tool_llm_breakdown": [],
                }
            ],
        },
    )

    assert result == "507f1f77bcf86cd799439021"
    assert collection.last_inserted is not None
    assert collection.last_inserted["message_uuid"] == "msg-1"
    assert collection.last_inserted["conversation_id"] == ObjectId("507f1f77bcf86cd799439011")
    assert collection.last_inserted["execution_trace"][0]["node_name"] == "chat_agent"
    assert collection.last_inserted["token_usage_detail"]["is_complete"] is True
    assert isinstance(collection.last_inserted["created_at"], datetime.datetime)
    assert isinstance(collection.last_inserted["updated_at"], datetime.datetime)


def test_add_message_trace_rejects_legacy_token_usage_payload(monkeypatch):
    """验证 token_usage_detail 传旧结构时不会落库。"""

    collection = _DummyCollection()
    monkeypatch.setattr(
        service_module,
        "get_mongo_database",
        lambda: {"message_traces": collection},
    )
    monkeypatch.setattr(service_module, "_resolve_collection_name", lambda: "message_traces")

    result = service_module.add_message_trace(
        message_uuid="msg-1",
        conversation_id="507f1f77bcf86cd799439011",
        token_usage_detail={
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    )

    assert result is None
    assert collection.last_inserted is None


def test_add_message_trace_skips_when_trace_and_detail_are_empty(monkeypatch):
    """验证 execution_trace 与明细都为空时跳过写入。"""

    collection = _DummyCollection()
    monkeypatch.setattr(
        service_module,
        "get_mongo_database",
        lambda: {"message_traces": collection},
    )
    monkeypatch.setattr(service_module, "_resolve_collection_name", lambda: "message_traces")

    result = service_module.add_message_trace(
        message_uuid="msg-1",
        conversation_id="507f1f77bcf86cd799439011",
    )

    assert result is None
    assert collection.last_inserted is None


def test_add_message_trace_rejects_invalid_conversation_id(monkeypatch):
    """验证非法 conversation_id 抛 BAD_REQUEST。"""

    collection = _DummyCollection()
    monkeypatch.setattr(
        service_module,
        "get_mongo_database",
        lambda: {"message_traces": collection},
    )
    monkeypatch.setattr(service_module, "_resolve_collection_name", lambda: "message_traces")

    with pytest.raises(ServiceException) as exc_info:
        service_module.add_message_trace(
            message_uuid="msg-1",
            conversation_id="invalid-object-id",
            execution_trace=[{"node_name": "chat_agent", "model_name": "qwen-max"}],
        )

    assert exc_info.value.code == ResponseCode.BAD_REQUEST


def test_get_message_trace_by_message_uuid_returns_typed_model(monkeypatch):
    """验证 get_message_trace_by_message_uuid：返回 MessageTraceDocument。"""

    collection = _DummyCollection()
    collection.find_one_result = {
        "_id": ObjectId("507f1f77bcf86cd799439022"),
        "message_uuid": "msg-1",
        "conversation_id": ObjectId("507f1f77bcf86cd799439011"),
        "execution_trace": [
            {
                "node_name": "chat_agent",
                "model_name": "qwen-max",
                "input_messages": [],
                "output_text": "ok",
                "tool_calls": [],
            }
        ],
        "token_usage_detail": {
            "is_complete": True,
            "node_breakdown": [],
        },
        "created_at": datetime.datetime(2026, 1, 1, 10, 0, 0),
        "updated_at": datetime.datetime(2026, 1, 1, 10, 0, 0),
    }
    monkeypatch.setattr(
        service_module,
        "get_mongo_database",
        lambda: {"message_traces": collection},
    )
    monkeypatch.setattr(service_module, "_resolve_collection_name", lambda: "message_traces")

    result = service_module.get_message_trace_by_message_uuid(message_uuid="msg-1")

    assert isinstance(result, MessageTraceDocument)
    assert result is not None
    assert result.id == "507f1f77bcf86cd799439022"
    assert result.message_uuid == "msg-1"
    assert result.conversation_id == "507f1f77bcf86cd799439011"
    assert result.execution_trace is not None
    assert result.execution_trace[0].node_name == "chat_agent"
