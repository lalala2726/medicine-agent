import datetime

import pytest
from bson import ObjectId

from app.core.codes import ResponseCode
from app.core.exceptions import ServiceException
from app.schemas.conversation_summary import ConversationSummary
from app.services import conversation_summary_service as service_module


class _DummyCollection:
    def __init__(self):
        self.last_find_one_and_update_query: dict | None = None
        self.last_find_one_and_update_doc: dict | None = None
        self.last_find_one_query: dict | None = None
        self.find_one_and_update_result: dict | None = None
        self.find_one_result: dict | None = None

    def find_one_and_update(
            self,
            query: dict,
            update_doc: dict,
            upsert: bool,
            return_document: object,
    ) -> dict | None:
        self.last_find_one_and_update_query = query
        self.last_find_one_and_update_doc = update_doc
        _ = upsert
        _ = return_document
        return self.find_one_and_update_result

    def find_one(self, query: dict) -> dict | None:
        self.last_find_one_query = query
        return self.find_one_result


def test_save_conversation_summary_upserts_and_returns_id(monkeypatch):
    """验证 save_conversation_summary：按会话 upsert 并返回文档ID。"""

    collection = _DummyCollection()
    collection.find_one_and_update_result = {"_id": ObjectId("507f1f77bcf86cd799439031")}
    monkeypatch.setattr(
        service_module,
        "get_mongo_database",
        lambda: {"conversation_summaries": collection},
    )
    monkeypatch.setattr(service_module, "_resolve_collection_name", lambda: "conversation_summaries")

    result = service_module.save_conversation_summary(
        conversation_id="507f1f77bcf86cd799439011",
        summary_content="这是总结",
        summarized_messages=["msg-1", "msg-2", ""],
        status="success",
    )

    assert result == "507f1f77bcf86cd799439031"
    assert collection.last_find_one_and_update_query == {
        "conversation_id": ObjectId("507f1f77bcf86cd799439011"),
    }
    assert collection.last_find_one_and_update_doc is not None
    assert collection.last_find_one_and_update_doc["$set"]["summary_content"] == "这是总结"
    assert collection.last_find_one_and_update_doc["$set"]["summarized_messages"] == ["msg-1", "msg-2"]
    assert collection.last_find_one_and_update_doc["$set"]["status"] == "success"
    assert isinstance(collection.last_find_one_and_update_doc["$set"]["updated_at"], datetime.datetime)
    assert isinstance(collection.last_find_one_and_update_doc["$setOnInsert"]["created_at"], datetime.datetime)


def test_save_conversation_summary_rejects_invalid_conversation_id(monkeypatch):
    """验证非法 conversation_id 抛 BAD_REQUEST。"""

    collection = _DummyCollection()
    monkeypatch.setattr(
        service_module,
        "get_mongo_database",
        lambda: {"conversation_summaries": collection},
    )
    monkeypatch.setattr(service_module, "_resolve_collection_name", lambda: "conversation_summaries")

    with pytest.raises(ServiceException) as exc_info:
        service_module.save_conversation_summary(
            conversation_id="invalid-object-id",
            summary_content="summary",
            summarized_messages=[],
            status="success",
        )

    assert exc_info.value.code == ResponseCode.BAD_REQUEST


def test_get_conversation_summary_returns_typed_model(monkeypatch):
    """验证 get_conversation_summary：查询结果会转换为 ConversationSummary。"""

    collection = _DummyCollection()
    collection.find_one_result = {
        "_id": ObjectId("507f1f77bcf86cd799439032"),
        "conversation_id": ObjectId("507f1f77bcf86cd799439011"),
        "summary_content": "总结内容",
        "summarized_messages": ["msg-1"],
        "created_at": datetime.datetime(2026, 1, 1, 10, 0, 0),
        "updated_at": datetime.datetime(2026, 1, 1, 11, 0, 0),
        "status": "success",
    }
    monkeypatch.setattr(
        service_module,
        "get_mongo_database",
        lambda: {"conversation_summaries": collection},
    )
    monkeypatch.setattr(service_module, "_resolve_collection_name", lambda: "conversation_summaries")

    result = service_module.get_conversation_summary(
        conversation_id="507f1f77bcf86cd799439011",
    )

    assert isinstance(result, ConversationSummary)
    assert result is not None
    assert result.id == "507f1f77bcf86cd799439032"
    assert result.conversation_id == "507f1f77bcf86cd799439011"
    assert result.summary_content == "总结内容"
    assert result.summarized_messages == ["msg-1"]
    assert result.status == "success"
