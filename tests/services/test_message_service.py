import datetime

import pytest
from bson import ObjectId
from langchain_core.messages import AIMessage, HumanMessage

from app.core.codes import ResponseCode
from app.core.exceptions import ServiceException
from app.schemas.admin_message import AdminMessageDocument, MessageRole, MessageStatus
from app.services import message_service as service_module


class _DummyInsertResult:
    def __init__(self, inserted_id: str):
        self.inserted_id = inserted_id


class _DummyCursor:
    def __init__(self, rows: list[dict]):
        self._rows = list(rows)
        self.sort_field: str | None = None
        self.sort_direction: int | None = None
        self.skip_value: int = 0
        self.limit_value: int | None = None

    def sort(self, field: str, direction: int):
        self.sort_field = field
        self.sort_direction = direction
        return self

    def limit(self, value: int):
        self.limit_value = value
        return self

    def skip(self, value: int):
        self.skip_value = value
        return self

    def __iter__(self):
        rows = list(self._rows)
        if self.sort_field is not None:
            rows.sort(
                key=lambda item: item.get(self.sort_field),
                reverse=self.sort_direction == -1,
            )
        if self.skip_value:
            rows = rows[self.skip_value:]
        if self.limit_value is not None:
            rows = rows[: self.limit_value]
        return iter(rows)


class _DummyCollection:
    def __init__(self):
        self.last_inserted: dict | None = None
        self.last_find_query: dict | None = None
        self.last_find_one_query: dict | None = None
        self.find_rows: list[dict] = []
        self.find_one_result: dict | None = None

    def insert_one(self, document: dict) -> _DummyInsertResult:
        self.last_inserted = document
        return _DummyInsertResult("507f1f77bcf86cd799439011")

    def find(self, query: dict) -> _DummyCursor:
        self.last_find_query = query
        return _DummyCursor(self.find_rows)

    def find_one(self, query: dict) -> dict | None:
        self.last_find_one_query = query
        return self.find_one_result


def test_add_message_inserts_expected_document(monkeypatch):
    """验证 add_message：user 消息入库成功。"""

    collection = _DummyCollection()
    monkeypatch.setattr(
        service_module,
        "get_mongo_database",
        lambda: {"admin_messages": collection},
    )
    monkeypatch.setattr(service_module, "_resolve_collection_name", lambda: "admin_messages")

    result = service_module.add_message(
        conversation_id="507f1f77bcf86cd799439011",
        role=MessageRole.USER,
        content="你好",
        thought_chain=[{"step": "plan"}],
        message_uuid="msg-uuid-1",
    )

    assert result == "507f1f77bcf86cd799439011"
    assert collection.last_inserted is not None
    assert collection.last_inserted["uuid"] == "msg-uuid-1"
    assert collection.last_inserted["role"] == MessageRole.USER
    assert collection.last_inserted["status"] == MessageStatus.SUCCESS
    assert collection.last_inserted["content"] == "你好"
    assert collection.last_inserted["conversation_id"] == ObjectId("507f1f77bcf86cd799439011")
    assert "token_usage" not in collection.last_inserted
    assert isinstance(collection.last_inserted["created_at"], datetime.datetime)
    assert isinstance(collection.last_inserted["updated_at"], datetime.datetime)


def test_get_message_by_uuid_returns_typed_model(monkeypatch):
    """验证 get_message_by_uuid：查询结果会被转换为 AdminMessageDocument。"""

    collection = _DummyCollection()
    collection.find_one_result = {
        "_id": ObjectId("507f1f77bcf86cd799439012"),
        "uuid": "msg-1",
        "conversation_id": ObjectId("507f1f77bcf86cd799439011"),
        "role": "assistant",
        "content": "hello",
        "created_at": datetime.datetime(2026, 1, 1, 10, 0, 0),
        "updated_at": datetime.datetime(2026, 1, 1, 10, 0, 0),
    }
    monkeypatch.setattr(
        service_module,
        "get_mongo_database",
        lambda: {"admin_messages": collection},
    )
    monkeypatch.setattr(service_module, "_resolve_collection_name", lambda: "admin_messages")

    result = service_module.get_message_by_uuid(message_uuid="msg-1")

    assert isinstance(result, AdminMessageDocument)
    assert result is not None
    assert result.id == "507f1f77bcf86cd799439012"
    assert result.conversation_id == "507f1f77bcf86cd799439011"
    assert result.role == MessageRole.ASSISTANT


def test_list_messages_returns_typed_models(monkeypatch):
    """验证 list_messages：按会话查询并返回类型化消息列表。"""

    collection = _DummyCollection()
    collection.find_rows = [
        {
            "_id": ObjectId("507f1f77bcf86cd799439022"),
            "uuid": "msg-2",
            "conversation_id": ObjectId("507f1f77bcf86cd799439011"),
            "role": "assistant",
            "content": "b",
            "created_at": datetime.datetime(2026, 1, 1, 10, 0, 2),
            "updated_at": datetime.datetime(2026, 1, 1, 10, 0, 2),
        },
        {
            "_id": ObjectId("507f1f77bcf86cd799439021"),
            "uuid": "msg-1",
            "conversation_id": ObjectId("507f1f77bcf86cd799439011"),
            "role": "user",
            "content": "a",
            "created_at": datetime.datetime(2026, 1, 1, 10, 0, 1),
            "updated_at": datetime.datetime(2026, 1, 1, 10, 0, 1),
        },
    ]
    monkeypatch.setattr(
        service_module,
        "get_mongo_database",
        lambda: {"admin_messages": collection},
    )
    monkeypatch.setattr(service_module, "_resolve_collection_name", lambda: "admin_messages")

    result = service_module.list_messages(
        conversation_id="507f1f77bcf86cd799439011",
        limit=2,
        skip=0,
        ascending=True,
    )

    assert collection.last_find_query == {
        "conversation_id": ObjectId("507f1f77bcf86cd799439011"),
    }
    assert len(result) == 2
    assert all(isinstance(item, AdminMessageDocument) for item in result)
    assert result[0].content == "a"
    assert result[1].content == "b"


def test_list_messages_supports_skip_with_descending_order(monkeypatch):
    """验证 list_messages：支持 skip 分页，且可按 created_at 倒序读取。"""

    collection = _DummyCollection()
    collection.find_rows = [
        {
            "_id": ObjectId("507f1f77bcf86cd799439020"),
            "uuid": "msg-1",
            "conversation_id": ObjectId("507f1f77bcf86cd799439011"),
            "role": "assistant",
            "content": "a",
            "created_at": datetime.datetime(2026, 1, 1, 10, 0, 1),
            "updated_at": datetime.datetime(2026, 1, 1, 10, 0, 1),
        },
        {
            "_id": ObjectId("507f1f77bcf86cd799439021"),
            "uuid": "msg-2",
            "conversation_id": ObjectId("507f1f77bcf86cd799439011"),
            "role": "assistant",
            "content": "b",
            "created_at": datetime.datetime(2026, 1, 1, 10, 0, 2),
            "updated_at": datetime.datetime(2026, 1, 1, 10, 0, 2),
        },
        {
            "_id": ObjectId("507f1f77bcf86cd799439022"),
            "uuid": "msg-3",
            "conversation_id": ObjectId("507f1f77bcf86cd799439011"),
            "role": "assistant",
            "content": "c",
            "created_at": datetime.datetime(2026, 1, 1, 10, 0, 3),
            "updated_at": datetime.datetime(2026, 1, 1, 10, 0, 3),
        },
    ]
    monkeypatch.setattr(
        service_module,
        "get_mongo_database",
        lambda: {"admin_messages": collection},
    )
    monkeypatch.setattr(service_module, "_resolve_collection_name", lambda: "admin_messages")

    result = service_module.list_messages(
        conversation_id="507f1f77bcf86cd799439011",
        limit=1,
        skip=1,
        ascending=False,
    )

    assert len(result) == 1
    assert result[0].uuid == "msg-2"


def test_add_message_rejects_invalid_conversation_id(monkeypatch):
    """验证 add_message：非法 conversation_id 会返回 BAD_REQUEST。"""

    collection = _DummyCollection()
    monkeypatch.setattr(
        service_module,
        "get_mongo_database",
        lambda: {"admin_messages": collection},
    )
    monkeypatch.setattr(service_module, "_resolve_collection_name", lambda: "admin_messages")

    with pytest.raises(ServiceException) as exc_info:
        service_module.add_message(
            conversation_id="invalid-object-id",
            role=MessageRole.ASSISTANT,
            content="hello",
        )

    assert exc_info.value.code == ResponseCode.BAD_REQUEST


def test_add_message_persists_execution_trace_for_assistant(monkeypatch):
    """测试目标：assistant execution_trace 可落库；成功标准：trace 字段完整写入。"""

    collection = _DummyCollection()
    monkeypatch.setattr(service_module, "get_mongo_database", lambda: {"admin_messages": collection})
    monkeypatch.setattr(service_module, "_resolve_collection_name", lambda: "admin_messages")

    service_module.add_message(
        conversation_id="507f1f77bcf86cd799439011",
        role=MessageRole.ASSISTANT,
        content="助手回复",
        execution_trace=[
            {
                "node_name": "order_agent",
                "model_name": "qwen3-max",
                "input_messages": [{"role": "system", "content": "请处理订单"}],
                "output_text": "订单已处理",
                "tool_calls": [
                    {
                        "tool_name": "get_order_list",
                        "tool_input": {"page_num": 1},
                        "is_error": False,
                        "error_message": None,
                    }
                ],
            }
        ],
    )

    assert collection.last_inserted is not None
    execution_trace = collection.last_inserted["execution_trace"]
    assert execution_trace[0]["node_name"] == "order_agent"
    assert execution_trace[0]["model_name"] == "qwen3-max"
    assert execution_trace[0]["tool_calls"][0]["tool_name"] == "get_order_list"
    assert execution_trace[0]["tool_calls"][0]["tool_input"] == {"page_num": 1}
    assert "tool_output" not in execution_trace[0]["tool_calls"][0]


def test_add_message_persists_error_status(monkeypatch):
    """验证 add_message：支持显式写入 error 状态。"""

    collection = _DummyCollection()
    monkeypatch.setattr(service_module, "get_mongo_database", lambda: {"admin_messages": collection})
    monkeypatch.setattr(service_module, "_resolve_collection_name", lambda: "admin_messages")

    service_module.add_message(
        conversation_id="507f1f77bcf86cd799439011",
        role=MessageRole.ASSISTANT,
        status=MessageStatus.ERROR,
        content="查询失败",
    )

    assert collection.last_inserted is not None
    assert collection.last_inserted["status"] == MessageStatus.ERROR


def test_add_message_user_does_not_persist_execution_trace_by_default(monkeypatch):
    """测试目标：user 消息默认不带 execution_trace。"""

    collection = _DummyCollection()
    monkeypatch.setattr(service_module, "get_mongo_database", lambda: {"admin_messages": collection})
    monkeypatch.setattr(service_module, "_resolve_collection_name", lambda: "admin_messages")

    service_module.add_message(
        conversation_id="507f1f77bcf86cd799439011",
        role=MessageRole.USER,
        content="你好",
    )

    assert collection.last_inserted is not None
    assert "token_usage" not in collection.last_inserted
    assert collection.last_inserted.get("execution_trace") is None


def test_add_message_persists_token_usage_only_for_assistant(monkeypatch):
    """测试目标：仅 assistant 消息保存 token_usage。"""

    collection = _DummyCollection()
    monkeypatch.setattr(service_module, "get_mongo_database", lambda: {"admin_messages": collection})
    monkeypatch.setattr(service_module, "_resolve_collection_name", lambda: "admin_messages")

    service_module.add_message(
        conversation_id="507f1f77bcf86cd799439011",
        role=MessageRole.ASSISTANT,
        content="助手回复",
        token_usage={
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "total_tokens": 8,
            "is_complete": True,
            "node_breakdown": [],
        },
    )
    assert collection.last_inserted is not None
    assert collection.last_inserted["token_usage"]["total_tokens"] == 8

    service_module.add_message(
        conversation_id="507f1f77bcf86cd799439011",
        role=MessageRole.USER,
        content="用户提问",
        token_usage={
            "prompt_tokens": 9,
            "completion_tokens": 9,
            "total_tokens": 18,
            "is_complete": True,
            "node_breakdown": [],
        },
    )
    assert collection.last_inserted is not None
    assert "token_usage" not in collection.last_inserted


def test_get_history_maps_role_to_langchain_messages(monkeypatch):
    """验证 get_history：能按 role 正确映射为 HumanMessage/AIMessage。"""

    collection = _DummyCollection()
    collection.find_rows = [
        {
            "_id": ObjectId("507f1f77bcf86cd799439021"),
            "uuid": "msg-1",
            "conversation_id": ObjectId("507f1f77bcf86cd799439011"),
            "role": "user",
            "content": "用户提问",
            "created_at": datetime.datetime(2026, 1, 1, 10, 0, 1),
            "updated_at": datetime.datetime(2026, 1, 1, 10, 0, 1),
        },
        {
            "_id": ObjectId("507f1f77bcf86cd799439022"),
            "uuid": "msg-2",
            "conversation_id": ObjectId("507f1f77bcf86cd799439011"),
            "role": "assistant",
            "content": "助手回答",
            "created_at": datetime.datetime(2026, 1, 1, 10, 0, 2),
            "updated_at": datetime.datetime(2026, 1, 1, 10, 0, 2),
        },
    ]
    monkeypatch.setattr(
        service_module,
        "get_mongo_database",
        lambda: {"admin_messages": collection},
    )
    monkeypatch.setattr(service_module, "_resolve_collection_name", lambda: "admin_messages")

    result = service_module.get_history(
        conversation_id="507f1f77bcf86cd799439011",
        limit=10,
        ascending=True,
    )

    assert len(result) == 2
    assert isinstance(result[0], HumanMessage)
    assert isinstance(result[1], AIMessage)
    assert result[0].content == "用户提问"
    assert result[1].content == "助手回答"
