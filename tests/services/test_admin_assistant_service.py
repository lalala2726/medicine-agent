import asyncio
import datetime
import json

import pytest
from bson import ObjectId
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage

from app.core.codes import ResponseCode
from app.core.exceptions import ServiceException
from app.schemas.document.message import MessageRole, MessageStatus
from app.schemas.document.conversation import ConversationDocument, ConversationListItem, ConversationType
from app.schemas.base_request import PageRequest
from app.schemas.sse_response import MessageType
from app.services import admin_assistant_service as service_module
from app.services.assistant_stream_service import AssistantStreamConfig


class _DummyGraph:
    def __init__(self, final_state: dict | None = None):
        self.final_state = final_state or {}
        self.captured_config = None

    def invoke(self, _state: dict, config: dict | None = None) -> dict:
        self.captured_config = config
        return self.final_state


def test_invoke_admin_workflow_passes_langsmith_config(monkeypatch):
    """测试目标：校验 workflow invoke 透传 tracing 配置；成功标准：graph.invoke 收到 config。"""

    graph = _DummyGraph(final_state={"results": {"chat": {"content": "ok"}}})
    monkeypatch.setattr(service_module, "ADMIN_WORKFLOW", graph)
    monkeypatch.setattr(
        service_module,
        "build_langsmith_runnable_config",
        lambda **_kwargs: {
            "run_name": "admin_assistant_graph",
            "tags": ["admin-assistant", "langgraph"],
            "metadata": {"entrypoint": "api.admin_assistant.chat"},
        },
    )

    result = service_module._invoke_admin_workflow({"user_input": "hello"})

    assert result["results"]["chat"]["content"] == "ok"
    assert graph.captured_config is not None
    assert graph.captured_config["run_name"] == "admin_assistant_graph"


def test_chat_list_returns_current_user_conversations(monkeypatch):
    """测试目标：会话列表查询透传分页参数；成功标准：仅返回会话 UUID 与标题列表。"""

    captured: dict = {}

    monkeypatch.setattr(service_module, "get_user_id", lambda: 100)
    monkeypatch.setattr(
        service_module,
        "list_admin_conversations",
        lambda *, user_id, page_num, page_size: (
            captured.update(
                {
                    "user_id": user_id,
                    "page_num": page_num,
                    "page_size": page_size,
                }
            ),
            ([ConversationListItem(conversation_uuid="conv-1", title="标题1")], 1),
        )[-1],
    )

    rows, total = service_module.conversation_list(
        page_request=PageRequest(
            page_num=2,
            page_size=20,
        )
    )

    assert captured == {
        "user_id": 100,
        "page_num": 2,
        "page_size": 20,
    }
    assert len(rows) == 1
    assert rows[0].conversation_uuid == "conv-1"
    assert rows[0].title == "标题1"
    assert total == 1


def test_delete_conversation_calls_repository_with_current_user(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(service_module, "get_user_id", lambda: 101)
    monkeypatch.setattr(
        service_module,
        "delete_admin_conversation",
        lambda *, conversation_uuid, user_id: (
            captured.update({"conversation_uuid": conversation_uuid, "user_id": user_id}),
            True,
        )[-1],
    )

    service_module.delete_conversation(conversation_uuid="conv-1")

    assert captured == {"conversation_uuid": "conv-1", "user_id": 101}


def test_delete_conversation_raises_not_found_when_missing(monkeypatch):
    monkeypatch.setattr(service_module, "get_user_id", lambda: 101)
    monkeypatch.setattr(
        service_module,
        "delete_admin_conversation",
        lambda **_kwargs: False,
    )

    with pytest.raises(ServiceException) as exc_info:
        service_module.delete_conversation(conversation_uuid="missing-conv")
    assert exc_info.value.code == ResponseCode.NOT_FOUND.code


def test_update_conversation_title_returns_normalized_title(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(service_module, "get_user_id", lambda: 101)
    monkeypatch.setattr(
        service_module,
        "update_admin_conversation_title",
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
        conversation_uuid="conv-1",
        title="  新标题  ",
    )

    assert title == "新标题"
    assert captured == {
        "conversation_uuid": "conv-1",
        "user_id": 101,
        "title": "新标题",
    }


def test_update_conversation_title_rejects_blank_title(monkeypatch):
    with pytest.raises(ServiceException) as exc_info:
        service_module.update_conversation_title(
            conversation_uuid="conv-1",
            title="   ",
        )
    assert exc_info.value.code == ResponseCode.BAD_REQUEST.code


def test_load_admin_conversation_returns_document_id(monkeypatch):
    monkeypatch.setattr(
        service_module,
        "get_admin_conversation",
        lambda *, conversation_uuid, user_id: ConversationDocument(
            _id=ObjectId("507f1f77bcf86cd799439041"),
            uuid=conversation_uuid,
            conversation_type=ConversationType.ADMIN,
            user_id=user_id,
            title="会话标题",
            create_time=datetime.datetime(2026, 1, 1, 10, 0, 0),
            update_time=datetime.datetime(2026, 1, 1, 10, 0, 0),
            is_deleted=0,
        ),
    )

    conversation_id = service_module._load_admin_conversation(
        conversation_uuid="conv-1",
        user_id=101,
    )

    assert conversation_id == "507f1f77bcf86cd799439041"


def test_load_admin_conversation_raises_database_error_when_id_missing(monkeypatch):
    monkeypatch.setattr(
        service_module,
        "get_admin_conversation",
        lambda *, conversation_uuid, user_id: ConversationDocument(
            uuid=conversation_uuid,
            conversation_type=ConversationType.ADMIN,
            user_id=user_id,
            title="会话标题",
            create_time=datetime.datetime(2026, 1, 1, 10, 0, 0),
            update_time=datetime.datetime(2026, 1, 1, 10, 0, 0),
            is_deleted=0,
        ),
    )

    with pytest.raises(ServiceException) as exc_info:
        service_module._load_admin_conversation(
            conversation_uuid="conv-1",
            user_id=101,
        )
    assert exc_info.value.code == ResponseCode.DATABASE_ERROR.code


def test_prepare_new_conversation_returns_context_with_created_event(monkeypatch):
    """测试目标：新会话上下文正确构建；成功标准：包含会话创建事件与首轮问题历史。"""

    scheduled_title_calls: list[dict] = []

    monkeypatch.setattr(service_module.uuid, "uuid4", lambda: "new-conv-uuid")
    monkeypatch.setattr(
        service_module,
        "add_admin_conversation",
        lambda *, conversation_uuid, user_id: f"db-{conversation_uuid}-{user_id}",
    )
    monkeypatch.setattr(
        service_module,
        "_schedule_title_generation",
        lambda **kwargs: scheduled_title_calls.append(kwargs),
    )

    context = service_module._prepare_new_conversation(
        question="新建会话",
        user_id=100,
    )

    assert isinstance(context, service_module.ConversationContext)
    assert context.conversation_uuid == "new-conv-uuid"
    assert context.conversation_id == "db-new-conv-uuid-100"
    assert [message.content for message in context.history_messages] == ["新建会话"]
    assert isinstance(context.history_messages[0], HumanMessage)
    assert context.is_new_conversation is True
    assert len(context.initial_emitted_events) == 1
    session_event = context.initial_emitted_events[0]
    assert session_event.type == MessageType.NOTICE
    assert session_event.content.state == "created"
    assert session_event.meta == {"conversation_uuid": "new-conv-uuid"}
    assert scheduled_title_calls == [{"conversation_uuid": "new-conv-uuid", "question": "新建会话"}]


def test_prepare_existing_conversation_returns_context_with_history(monkeypatch):
    """测试目标：旧会话上下文正确构建；成功标准：加载会话并返回历史窗口。"""

    captured: dict = {}
    expected_history = [HumanMessage(content="历史问题"), AIMessage(content="历史回答")]

    monkeypatch.setattr(
        service_module,
        "_load_admin_conversation",
        lambda *, conversation_uuid, user_id: (
            captured.update({"conversation_uuid": conversation_uuid, "user_id": user_id}),
            "507f1f77bcf86cd799439011",
        )[-1],
    )
    monkeypatch.setattr(
        service_module,
        "load_history",
        lambda *, conversation_id, limit: (
            captured.update({"conversation_id": conversation_id, "limit": limit}),
            expected_history,
        )[-1],
    )

    context = service_module._prepare_existing_conversation(
        conversation_uuid="conv-1",
        user_id=100,
        question="本轮问题",
    )

    assert isinstance(context, service_module.ConversationContext)
    assert context.conversation_uuid == "conv-1"
    assert context.conversation_id == "507f1f77bcf86cd799439011"
    assert [message.content for message in context.history_messages] == [
        "历史问题",
        "历史回答",
        "本轮问题",
    ]
    assert isinstance(context.history_messages[-1], HumanMessage)
    assert context.initial_emitted_events == ()
    assert context.is_new_conversation is False
    assert captured == {
        "conversation_uuid": "conv-1",
        "user_id": 100,
        "conversation_id": "507f1f77bcf86cd799439011",
        "limit": 50,
    }


def test_prepare_conversation_context_routes_by_conversation_uuid(monkeypatch):
    """测试目标：会话准备总入口分发正确；成功标准：按 UUID 是否为空路由到对应分支。"""

    call_order: list[tuple[str, dict]] = []
    new_context = service_module.ConversationContext(
        conversation_uuid="new-conv",
        conversation_id="new-id",
        history_messages=[],
        initial_emitted_events=(),
        is_new_conversation=True,
    )
    existing_context = service_module.ConversationContext(
        conversation_uuid="conv-1",
        conversation_id="old-id",
        history_messages=[HumanMessage(content="历史")],
        initial_emitted_events=(),
        is_new_conversation=False,
    )

    monkeypatch.setattr(
        service_module,
        "_prepare_new_conversation",
        lambda *, question, user_id: (
            call_order.append(("new", {"question": question, "user_id": user_id})),
            new_context,
        )[-1],
    )
    monkeypatch.setattr(
        service_module,
        "_prepare_existing_conversation",
        lambda *, conversation_uuid, user_id, question: (
            call_order.append(
                (
                    "existing",
                    {"conversation_uuid": conversation_uuid, "user_id": user_id, "question": question},
                )
            ),
            existing_context,
        )[-1],
    )

    context_new = service_module._prepare_conversation_context(
        question="问题A",
        user_id=100,
        conversation_uuid=None,
    )
    context_existing = service_module._prepare_conversation_context(
        question="问题B",
        user_id=101,
        conversation_uuid="conv-1",
    )

    assert context_new is new_context
    assert context_existing is existing_context
    assert call_order == [
        ("new", {"question": "问题A", "user_id": 100}),
        ("existing", {"conversation_uuid": "conv-1", "user_id": 101, "question": "问题B"}),
    ]


def test_assistant_chat_schedules_user_persist_without_blocking_main_flow(monkeypatch):
    """测试目标：user 消息走后台调度；成功标准：主流程只触发调度，不依赖同步落库。"""

    captured: dict = {}
    background_calls: list[dict] = []
    saved_messages: list[dict] = []
    call_order: list[str] = []

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

    monkeypatch.setattr(service_module, "create_streaming_response", _fake_create_streaming_response)
    monkeypatch.setattr(service_module, "get_user_id", lambda: 100)
    monkeypatch.setattr(
        service_module,
        "get_admin_conversation",
        lambda *, conversation_uuid, user_id: ConversationDocument(
            _id=ObjectId("507f1f77bcf86cd799439011"),
            uuid=conversation_uuid,
            conversation_type=ConversationType.ADMIN,
            user_id=user_id,
            title="会话标题",
            create_time=datetime.datetime(2026, 1, 1, 10, 0, 0),
            update_time=datetime.datetime(2026, 1, 1, 10, 0, 0),
            is_deleted=0,
        ),
    )
    monkeypatch.setattr(
        service_module,
        "load_history",
        lambda **_kwargs: (
            call_order.append("load_history"),
            [HumanMessage(content="历史问题"), AIMessage(content="历史回答")],
        )[-1],
    )
    monkeypatch.setattr(
        service_module,
        "add_message",
        lambda **kwargs: (
            saved_messages.append(kwargs),
            call_order.append(f"add_{kwargs['role']}"),
            "507f1f77bcf86cd799439012",
        )[-1],
    )
    monkeypatch.setattr(
        service_module,
        "_schedule_background_task",
        lambda *, task_name, func, kwargs: (
            background_calls.append({"task_name": task_name, "kwargs": kwargs}),
            func(**kwargs),
        )[-1],
    )

    response = service_module.assistant_chat(question="代理测试", conversation_uuid="conv-1")

    assert isinstance(response, StreamingResponse)
    assert captured["question"] == "代理测试"
    stream_config = captured["config"]
    initial_state = stream_config.build_initial_state("x")
    assert "context" in initial_state
    assert "messages" in initial_state
    assert [message.content for message in initial_state["history_messages"]] == [
        "历史问题",
        "历史回答",
        "代理测试",
    ]
    assert [message.content for message in initial_state["messages"]] == [
        "历史问题",
        "历史回答",
        "代理测试",
    ]
    assert call_order == ["load_history", "add_user"]
    assert background_calls == [
        {
            "task_name": "persist_user_message",
            "kwargs": {
                "conversation_id": "507f1f77bcf86cd799439011",
                "question": "代理测试",
            },
        }
    ]
    assert saved_messages == [
        {
            "conversation_id": "507f1f77bcf86cd799439011",
            "role": "user",
            "content": "代理测试",
        }
    ]


def test_answer_completed_schedules_async_persist_with_execution_trace(monkeypatch):
    """测试目标：assistant 完成回调仅调度后台任务；成功标准：调度参数含 execution_trace。"""

    background_calls: list[dict] = []
    saved_messages: list[dict] = []
    saved_traces: list[dict] = []
    resolve_calls: list[dict] = []

    monkeypatch.setattr(
        service_module,
        "_schedule_background_task",
        lambda *, task_name, func, kwargs: (
            background_calls.append({"task_name": task_name, "kwargs": kwargs}),
            func(**kwargs),
        )[-1],
    )
    monkeypatch.setattr(
        service_module,
        "add_message",
        lambda **kwargs: saved_messages.append(kwargs) or "507f1f77bcf86cd799439012",
    )
    monkeypatch.setattr(
        service_module,
        "add_message_trace",
        lambda **kwargs: saved_traces.append(kwargs) or "507f1f77bcf86cd799439013",
    )
    monkeypatch.setattr(service_module.uuid, "uuid4", lambda: "msg-uuid-1")
    monkeypatch.setattr(
        service_module,
        "resolve_persistable_token_usage",
        lambda token_usage, execution_trace: (
            resolve_calls.append({"token_usage": token_usage, "execution_trace": execution_trace}),
            {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
                "is_complete": True,
                "node_breakdown": [],
            },
        )[-1],
    )

    callback = service_module._build_assistant_message_callback(
        conversation_id="507f1f77bcf86cd799439011",
    )
    asyncio.run(
        callback(
            "AI回复",
            [
                {
                    "node_name": "chat_agent",
                    "model_name": "qwen-max",
                    "input_messages": [{"role": "human", "content": "用户问题"}],
                    "output_text": "AI回复",
                    "tool_calls": [
                        {
                            "tool_name": "query_orders",
                            "tool_input": {"limit": 10},
                            "is_error": False,
                            "error_message": None,
                        }
                    ],
                }
            ],
            False,
        )
    )

    assert len(background_calls) == 1
    assert background_calls[0]["task_name"] == "persist_assistant_message"
    assert background_calls[0]["kwargs"]["conversation_id"] == "507f1f77bcf86cd799439011"
    assert background_calls[0]["kwargs"]["answer_text"] == "AI回复"
    assert background_calls[0]["kwargs"]["status"] == MessageStatus.SUCCESS
    assert saved_messages[-1]["status"] == MessageStatus.SUCCESS
    assert saved_messages[-1]["message_uuid"] == "msg-uuid-1"
    assert saved_messages[-1]["token_usage"]["total_tokens"] == 2
    assert "execution_trace" not in saved_messages[-1]
    assert saved_traces[-1]["message_uuid"] == "msg-uuid-1"
    assert saved_traces[-1]["execution_trace"][0]["node_name"] == "chat_agent"
    assert saved_traces[-1]["token_usage_detail"]["is_complete"] is True
    assert resolve_calls and resolve_calls[-1]["token_usage"] is None


def test_answer_completed_marks_error_status_when_stream_failed(monkeypatch):
    """测试目标：流式执行报错时，assistant 消息落库状态为 error。"""

    saved_messages: list[dict] = []
    saved_traces: list[dict] = []

    monkeypatch.setattr(
        service_module,
        "_schedule_background_task",
        lambda *, task_name, func, kwargs: func(**kwargs),
    )
    monkeypatch.setattr(
        service_module,
        "add_message",
        lambda **kwargs: saved_messages.append(kwargs) or "507f1f77bcf86cd799439012",
    )
    monkeypatch.setattr(
        service_module,
        "add_message_trace",
        lambda **kwargs: saved_traces.append(kwargs) or None,
    )
    monkeypatch.setattr(service_module.uuid, "uuid4", lambda: "msg-uuid-error")
    monkeypatch.setattr(
        service_module,
        "resolve_persistable_token_usage",
        lambda _token_usage, _execution_trace: None,
    )

    callback = service_module._build_assistant_message_callback(
        conversation_id="507f1f77bcf86cd799439011",
    )
    asyncio.run(
        callback(
            "错误提示",
            [],
            True,
        )
    )

    assert saved_messages[-1]["status"] == MessageStatus.ERROR
    assert saved_messages[-1]["message_uuid"] == "msg-uuid-error"
    assert saved_traces[-1]["message_uuid"] == "msg-uuid-error"


def test_persist_assistant_message_trace_failure_only_logs_warning(monkeypatch):
    """测试目标：trace 持久化失败仅告警；成功标准：主消息已保存且无异常抛出。"""

    saved_messages: list[dict] = []
    warning_calls: list[dict] = []

    monkeypatch.setattr(
        service_module,
        "add_message",
        lambda **kwargs: saved_messages.append(kwargs) or "507f1f77bcf86cd799439012",
    )
    monkeypatch.setattr(
        service_module,
        "add_message_trace",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("trace failed")),
    )
    monkeypatch.setattr(
        service_module,
        "resolve_persistable_token_usage",
        lambda _token_usage, _execution_trace: {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
            "is_complete": True,
            "node_breakdown": [],
        },
    )
    monkeypatch.setattr(service_module.uuid, "uuid4", lambda: "msg-uuid-2")

    class _DummyLogger:
        def warning(self, message: str, **kwargs):
            warning_calls.append({"message": message, "kwargs": kwargs})

    monkeypatch.setattr(service_module.logger, "opt", lambda **_kwargs: _DummyLogger())

    service_module._persist_assistant_message(
        conversation_id="507f1f77bcf86cd799439011",
        answer_text="复杂回复",
        execution_trace=[
            {
                "node_name": "supervisor_agent",
                "model_name": "qwen-max",
                "input_messages": [],
                "output_text": "计划已生成",
                "tool_calls": [],
            }
        ],
        token_usage=None,
        status=MessageStatus.SUCCESS,
    )

    assert saved_messages[-1]["message_uuid"] == "msg-uuid-2"
    assert saved_messages[-1]["token_usage"]["total_tokens"] == 2
    assert warning_calls
    assert warning_calls[0]["kwargs"]["message_uuid"] == "msg-uuid-2"


def test_persist_failure_only_logs_warning(monkeypatch):
    """测试目标：后台任务失败仅日志；成功标准：异常不抛出且 warning 被记录。"""

    warning_calls: list[dict] = []

    class _DummyLogger:
        def warning(self, message: str, **kwargs):
            warning_calls.append({"message": message, "kwargs": kwargs})

    monkeypatch.setattr(service_module.logger, "opt", lambda **_kwargs: _DummyLogger())

    class _ImmediateThread:
        def __init__(self, target, daemon=True):
            self._target = target
            self.daemon = daemon

        def start(self):
            self._target()

    monkeypatch.setattr(service_module.threading, "Thread", _ImmediateThread)

    service_module._schedule_background_task(
        task_name="broken_task",
        func=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        kwargs={},
    )

    assert warning_calls
    assert warning_calls[0]["kwargs"]["task_name"] == "broken_task"


def test_assistant_chat_new_conversation_injects_created_session_event(monkeypatch):
    """测试目标：新会话路径注入创建事件；成功标准：事件正确且 user 消息经后台调度。"""

    captured: dict = {}
    scheduled_title_calls: list[dict] = []
    background_calls: list[dict] = []

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

    monkeypatch.setattr(service_module, "create_streaming_response", _fake_create_streaming_response)
    monkeypatch.setattr(service_module, "get_user_id", lambda: 100)
    monkeypatch.setattr(service_module.uuid, "uuid4", lambda: "new-conv-uuid")
    monkeypatch.setattr(
        service_module,
        "add_admin_conversation",
        lambda *, conversation_uuid, user_id: f"db-{conversation_uuid}-{user_id}",
    )
    monkeypatch.setattr(
        service_module,
        "_schedule_title_generation",
        lambda **kwargs: scheduled_title_calls.append(kwargs),
    )
    monkeypatch.setattr(
        service_module,
        "_schedule_background_task",
        lambda *, task_name, func, kwargs: background_calls.append({"task_name": task_name, "kwargs": kwargs}),
    )

    response = service_module.assistant_chat(question="新建会话")

    assert isinstance(response, StreamingResponse)
    assert captured["question"] == "新建会话"
    stream_config = captured["config"]
    assert len(stream_config.initial_emitted_events) == 1
    session_event = stream_config.initial_emitted_events[0]
    assert session_event.type == MessageType.NOTICE
    assert session_event.content.state == "created"
    assert session_event.meta == {"conversation_uuid": "new-conv-uuid"}
    assert scheduled_title_calls == [{"conversation_uuid": "new-conv-uuid", "question": "新建会话"}]
    assert background_calls[0]["task_name"] == "persist_user_message"


def test_load_history_reads_latest_window_and_returns_chronological(monkeypatch):
    """测试目标：历史读取顺序正确；成功标准：倒序读后返回正序窗口。"""

    captured: dict = {}

    def _fake_list_messages(*, conversation_id: str, limit: int, ascending: bool):
        captured["conversation_id"] = conversation_id
        captured["limit"] = limit
        captured["ascending"] = ascending
        return [
            type("Doc", (), {"role": MessageRole.USER, "content": "Q2"})(),
            type("Doc", (), {"role": MessageRole.AI, "content": "A1"})(),
            type("Doc", (), {"role": MessageRole.USER, "content": "Q1"})(),
        ]

    monkeypatch.setattr(service_module, "list_messages", _fake_list_messages)

    history_messages = service_module.load_history(
        conversation_id="507f1f77bcf86cd799439011",
        limit=50,
    )

    assert captured == {
        "conversation_id": "507f1f77bcf86cd799439011",
        "limit": 50,
        "ascending": False,
    }
    assert isinstance(history_messages[0], HumanMessage)
    assert isinstance(history_messages[1], AIMessage)
    assert isinstance(history_messages[2], HumanMessage)
    assert [message.content for message in history_messages] == ["Q1", "A1", "Q2"]


def test_conversation_messages_returns_latest_page_in_chronological_order(monkeypatch):
    """测试目标：历史消息按页读取最新窗口，返回时按时间升序。"""

    monkeypatch.setattr(service_module, "get_user_id", lambda: 100)
    monkeypatch.setattr(
        service_module,
        "_load_admin_conversation",
        lambda *, conversation_uuid, user_id: "507f1f77bcf86cd799439011",
    )
    mock_docs = [
        {
            "uuid": "msg-ai-2",
            "role": MessageRole.AI,
            "status": MessageStatus.SUCCESS,
            "content": "AI第二条",
        },
        {
            "uuid": "msg-user-1",
            "role": MessageRole.USER,
            "status": MessageStatus.SUCCESS,
            "content": "用户第一条",
        },
    ]
    monkeypatch.setattr(
        service_module,
        "list_messages",
        lambda **_kwargs: [
            type("Doc", (), item)() for item in mock_docs
        ],
    )

    result = service_module.conversation_messages(
        conversation_uuid="conv-1",
        page_request=PageRequest(page_num=1, page_size=50),
    )

    assert [item.id for item in result] == ["msg-user-1", "msg-ai-2"]
    assert result[0].role == "user"
    assert result[0].status is None
    assert result[1].role == "ai"
    assert result[1].status == "success"
    assert result[1].thought_chain is None


def test_should_stream_token_only_allows_chat_and_supervisor_nodes():
    """测试目标：仅 chat/supervisor 节点允许输出 token。"""

    latest_state = {"routing": {"route_target": "chat_agent", "task_difficulty": "simple"}}
    assert service_module._should_stream_token("chat_agent", latest_state) is True
    assert service_module._should_stream_token("supervisor_agent", latest_state) is True
    assert service_module._should_stream_token("order_agent", latest_state) is False
    assert service_module._should_stream_token("product_agent", latest_state) is False
