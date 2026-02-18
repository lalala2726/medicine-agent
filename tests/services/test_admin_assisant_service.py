import asyncio
import json

from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage

from app.schemas.sse_response import MessageType
from app.services import admin_assisant_service as service_module
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
        lambda *, conversation_uuid, user_id: {
            "_id": "507f1f77bcf86cd799439011",
            "uuid": conversation_uuid,
            "user_id": user_id,
        },
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
    assert stream_config.build_initial_state("x")["execution_traces"] == []
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
    """测试目标：assistant 完成回调仅调度后台任务；成功标准：调度参数含 usage 与 execution_trace。"""

    background_calls: list[dict] = []
    saved_messages: list[dict] = []
    merge_calls: list[dict] = []

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
        "merge_assistant_token_usage",
        lambda **kwargs: (
            merge_calls.append(kwargs),
            {
                "prompt_tokens": 8,
                "completion_tokens": 4,
                "total_tokens": 12,
                "breakdown": None,
            },
        )[-1],
    )
    monkeypatch.setattr(
        service_module,
        "add_message",
        lambda **kwargs: saved_messages.append(kwargs) or "507f1f77bcf86cd799439012",
    )

    callback = service_module._build_assistant_message_callback(
        conversation_id="507f1f77bcf86cd799439011",
        question="用户问题",
    )
    asyncio.run(
        callback(
            "AI回复",
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            [
                {
                    "node_name": "chat_agent",
                    "model_name": "qwen-max",
                    "input_messages": [{"role": "human", "content": "用户问题"}],
                    "output_text": "AI回复",
                    "tool_calls": [],
                }
            ],
        )
    )

    assert background_calls == [
        {
            "task_name": "persist_assistant_message",
            "kwargs": {
                "conversation_id": "507f1f77bcf86cd799439011",
                "question": "用户问题",
                "answer_text": "AI回复",
                "stream_token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                "execution_trace": [
                    {
                        "node_name": "chat_agent",
                        "model_name": "qwen-max",
                        "input_messages": [{"role": "human", "content": "用户问题"}],
                        "output_text": "AI回复",
                        "tool_calls": [],
                    }
                ],
            },
        }
    ]
    assert merge_calls == [
        {
            "stream_token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "prompt_text": "用户问题",
            "completion_text": "AI回复",
        }
    ]
    assert saved_messages[-1]["execution_trace"][0]["node_name"] == "chat_agent"


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
    assert session_event.type == MessageType.STATUS
    assert session_event.content.node == "conversation"
    assert session_event.content.state == "created"
    assert session_event.meta == {"conversation_uuid": "new-conv-uuid"}
    assert scheduled_title_calls == [{"conversation_uuid": "new-conv-uuid", "question": "新建会话"}]
    assert background_calls[0]["task_name"] == "persist_user_message"


def test_load_history_reads_latest_window_and_returns_chronological(monkeypatch):
    """测试目标：历史读取顺序正确；成功标准：倒序读后返回正序窗口。"""

    captured: dict = {}

    def _fake_get_history(*, conversation_id: str, limit: int, ascending: bool):
        captured["conversation_id"] = conversation_id
        captured["limit"] = limit
        captured["ascending"] = ascending
        return [HumanMessage(content="Q2"), AIMessage(content="A1"), HumanMessage(content="Q1")]

    monkeypatch.setattr(service_module, "get_history", _fake_get_history)

    history_messages = service_module.load_history(
        conversation_id="507f1f77bcf86cd799439011",
        limit=50,
    )

    assert captured == {
        "conversation_id": "507f1f77bcf86cd799439011",
        "limit": 50,
        "ascending": False,
    }
    assert [message.content for message in history_messages] == ["Q1", "A1", "Q2"]
