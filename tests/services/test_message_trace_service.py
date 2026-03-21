import datetime

import pytest
from bson import ObjectId

from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException
from app.core.llms.provider import LlmProvider
from app.schemas.document.message_trace import MessageTraceDocument, MessageTraceProvider
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
    """测试目的：message_trace 新结构可正常落库；预期结果：workflow/execution_trace/token_usage 均按新字段写入。"""

    collection = _DummyCollection()
    monkeypatch.setattr(
        service_module,
        "get_mongo_database",
        lambda: {"message_traces": collection},
    )
    monkeypatch.setattr(service_module, "_resolve_collection_name", lambda: "message_traces")
    monkeypatch.setattr(service_module, "resolve_provider", lambda _provider=None: LlmProvider.OPENAI)

    result = service_module.add_message_trace(
        message_uuid="msg-1",
        conversation_id="507f1f77bcf86cd799439011",
        execution_trace=[
            {
                "sequence": 1,
                "node_name": "chat_agent",
                "model_name": "qwen-max",
                "output_text": "AI回复",
                "tool_calls": [],
            }
        ],
        token_usage={
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
            "is_complete": True,
        },
    )

    assert result == "507f1f77bcf86cd799439021"
    assert collection.last_inserted is not None
    assert collection.last_inserted["message_uuid"] == "msg-1"
    assert collection.last_inserted["conversation_id"] == ObjectId("507f1f77bcf86cd799439011")
    assert collection.last_inserted["provider"] == MessageTraceProvider.OPENAI
    assert collection.last_inserted["workflow"]["workflow_name"] == "admin_assistant_graph"
    assert collection.last_inserted["execution_trace"][0]["node_name"] == "chat_agent"
    assert collection.last_inserted["token_usage"]["is_complete"] is True
    assert isinstance(collection.last_inserted["created_at"], datetime.datetime)
    assert isinstance(collection.last_inserted["updated_at"], datetime.datetime)


def test_add_message_trace_uses_env_provider_when_provider_missing(monkeypatch):
    """验证 add_message_trace：未显式传 provider 时按环境解析厂家。"""

    collection = _DummyCollection()
    monkeypatch.setattr(
        service_module,
        "get_mongo_database",
        lambda: {"message_traces": collection},
    )
    monkeypatch.setattr(service_module, "_resolve_collection_name", lambda: "message_traces")
    monkeypatch.setattr(service_module, "resolve_provider", lambda _provider=None: LlmProvider.ALIYUN)

    result = service_module.add_message_trace(
        message_uuid="msg-env-provider",
        conversation_id="507f1f77bcf86cd799439011",
        execution_trace=[{"node_name": "chat_agent", "model_name": "qwen-max"}],
    )

    assert result == "507f1f77bcf86cd799439021"
    assert collection.last_inserted is not None
    assert collection.last_inserted["provider"] == MessageTraceProvider.ALIYUN


def test_add_message_trace_supports_cancelled_workflow_status(monkeypatch):
    """验证 message_trace 支持 cancelled 工作流终态。"""

    collection = _DummyCollection()
    monkeypatch.setattr(
        service_module,
        "get_mongo_database",
        lambda: {"message_traces": collection},
    )
    monkeypatch.setattr(service_module, "_resolve_collection_name", lambda: "message_traces")
    monkeypatch.setattr(service_module, "resolve_provider", lambda _provider=None: LlmProvider.OPENAI)

    service_module.add_message_trace(
        message_uuid="msg-cancelled",
        conversation_id="507f1f77bcf86cd799439011",
        workflow_status="cancelled",
        execution_trace=[{"node_name": "chat_agent", "model_name": "gpt"}],
    )

    assert collection.last_inserted is not None
    assert collection.last_inserted["workflow"]["workflow_status"] == "cancelled"


def test_add_message_trace_allows_custom_workflow_name(monkeypatch):
    collection = _DummyCollection()
    monkeypatch.setattr(
        service_module,
        "get_mongo_database",
        lambda: {"message_traces": collection},
    )
    monkeypatch.setattr(service_module, "_resolve_collection_name", lambda: "message_traces")
    monkeypatch.setattr(service_module, "resolve_provider", lambda _provider=None: LlmProvider.OPENAI)

    result = service_module.add_message_trace(
        message_uuid="msg-client",
        conversation_id="507f1f77bcf86cd799439011",
        workflow_name="client_assistant_graph",
        execution_trace=[{"node_name": "chat_agent", "model_name": "gpt"}],
    )

    assert result == "507f1f77bcf86cd799439021"
    assert collection.last_inserted is not None
    assert collection.last_inserted["workflow"]["workflow_name"] == "client_assistant_graph"


def test_add_message_trace_explicit_provider_overrides_env(monkeypatch):
    """验证 add_message_trace：显式 provider 覆盖环境默认厂家。"""

    collection = _DummyCollection()
    monkeypatch.setattr(
        service_module,
        "get_mongo_database",
        lambda: {"message_traces": collection},
    )
    monkeypatch.setattr(service_module, "_resolve_collection_name", lambda: "message_traces")

    captured: dict[str, str | None] = {"provider": None}

    def _fake_resolve_provider(provider=None):
        captured["provider"] = str(provider)
        return LlmProvider.VOLCENGINE

    monkeypatch.setattr(service_module, "resolve_provider", _fake_resolve_provider)

    result = service_module.add_message_trace(
        message_uuid="msg-explicit-provider",
        conversation_id="507f1f77bcf86cd799439011",
        provider="volcengine",
        execution_trace=[{"node_name": "chat_agent", "model_name": "doubao-seed"}],
    )

    assert result == "507f1f77bcf86cd799439021"
    assert captured["provider"] == "volcengine"
    assert collection.last_inserted is not None
    assert collection.last_inserted["provider"] == MessageTraceProvider.VOLCENGINE


def test_add_message_trace_ignores_invalid_token_usage_payload(monkeypatch):
    """测试目的：非法 token_usage 不影响 trace 主体落库；预期结果：记录写入但不包含 token_usage 字段。"""

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
        token_usage={"invalid": "payload"},
    )

    assert result == "507f1f77bcf86cd799439021"
    assert collection.last_inserted is not None
    assert "token_usage" not in collection.last_inserted


def test_add_message_trace_skips_when_trace_and_detail_are_empty(monkeypatch):
    """测试目的：空轨迹也应落库；预期结果：返回文档ID且 execution_trace 为空数组。"""

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

    assert result == "507f1f77bcf86cd799439021"
    assert collection.last_inserted is not None
    assert collection.last_inserted["execution_trace"] == []


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
    """测试目的：读取接口按新 schema 反序列化；预期结果：workflow/sequence/token_usage 字段解析正确。"""

    collection = _DummyCollection()
    collection.find_one_result = {
        "_id": ObjectId("507f1f77bcf86cd799439022"),
        "message_uuid": "msg-1",
        "conversation_id": ObjectId("507f1f77bcf86cd799439011"),
        "provider": "aliyun",
        "workflow": {
            "workflow_name": "admin_assistant_graph",
            "workflow_status": "success",
            "execution_path": ["chat_agent"],
            "final_node": "chat_agent",
            "route_targets": [],
            "task_difficulty": None,
        },
        "execution_trace": [
            {
                "sequence": 1,
                "node_name": "chat_agent",
                "model_name": "qwen-max",
                "output_text": "ok",
                "tool_calls": [],
            }
        ],
        "token_usage": {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
            "is_complete": True,
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
    assert result.provider == MessageTraceProvider.ALIYUN
    assert result.execution_trace is not None
    assert result.workflow.workflow_status == "success"
    assert result.execution_trace[0].node_name == "chat_agent"
