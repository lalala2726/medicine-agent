import asyncio
import json

from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage

from app.services import admin_assisant_service as service_module
from app.schemas.sse_response import MessageType
from app.services.assistant_stream_service import AssistantStreamConfig


class _DummyGraph:
    def __init__(self, final_state: dict | None = None):
        self.final_state = final_state or {}
        self.captured_config = None

    def invoke(self, _state: dict, config: dict | None = None) -> dict:
        self.captured_config = config
        return self.final_state


def test_invoke_admin_workflow_passes_langsmith_config(monkeypatch):
    graph = _DummyGraph(final_state={"results": {"chat": {"content": "ok"}}})
    monkeypatch.setattr(service_module, "ADMIN_WORKFLOW", graph)
    monkeypatch.setattr(
        service_module,
        "build_langsmith_runnable_config",
        lambda **kwargs: {
            "run_name": "admin_assistant_graph",
            "tags": ["admin-assistant", "langgraph"],
            "metadata": {"entrypoint": "api.admin_assistant.chat"},
        },
    )

    state = {"user_input": "hello"}
    result = service_module._invoke_admin_workflow(state)

    assert result["results"]["chat"]["content"] == "ok"
    assert graph.captured_config is not None
    assert graph.captured_config["run_name"] == "admin_assistant_graph"


def test_assistant_chat_delegates_to_stream_service(monkeypatch):
    captured: dict = {}
    scheduled_calls: list[dict] = []
    saved_messages: list[dict] = []
    merge_usage_calls: list[dict] = []
    call_order: list[str] = []

    def _fake_create_streaming_response(question: str, config: AssistantStreamConfig):
        captured["question"] = question
        captured["config"] = config

        async def _stream():
            yield (
                "data: "
                + json.dumps(
                    {
                        "content": {"text": "delegated"},
                        "type": "answer",
                        "is_end": False,
                        "timestamp": 1,
                    },
                    ensure_ascii=False,
                )
                + "\n\n"
            )
            yield (
                "data: "
                + json.dumps(
                    {
                        "content": {"text": ""},
                        "type": "answer",
                        "is_end": True,
                        "timestamp": 2,
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
    monkeypatch.setattr(
        service_module,
        "_schedule_title_generation",
        lambda **kwargs: scheduled_calls.append(kwargs),
    )
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
        "add_message",
        lambda **kwargs: (
                saved_messages.append(kwargs),
                call_order.append(f"add_{kwargs['role']}"),
                "507f1f77bcf86cd799439012",
        )[-1],
    )
    monkeypatch.setattr(
        service_module,
        "load_history",
        lambda **kwargs: (
                call_order.append("load_history"),
                [
                    HumanMessage(content="历史问题"),
                    AIMessage(content="历史回答"),
                    HumanMessage(content="代理测试"),
                ],
        )[-1],
    )
    monkeypatch.setattr(
        service_module,
        "merge_assistant_token_usage",
        lambda **kwargs: (
                merge_usage_calls.append(kwargs),
                {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "intermediate_tokens": 3,
                    "total_tokens": 21,
                    "breakdown": [
                        {
                            "node_name": "chat_agent",
                            "prompt_tokens": 11,
                            "completion_tokens": 7,
                            "total_tokens": 18,
                        }
                    ],
                },
        )[-1],
    )

    response = service_module.assistant_chat(
        question="代理测试",
        conversation_uuid="conv-1",
    )

    assert isinstance(response, StreamingResponse)
    stream_config = captured["config"]
    assert captured["question"] == "代理测试"
    assert isinstance(stream_config, AssistantStreamConfig)
    assert stream_config.workflow is service_module.ADMIN_WORKFLOW
    assert stream_config.build_initial_state("x")["user_input"] == "x"
    history_messages = stream_config.build_initial_state("x")["history_messages"]
    assert len(history_messages) == 3
    assert isinstance(history_messages[0], HumanMessage)
    assert isinstance(history_messages[1], AIMessage)
    assert isinstance(history_messages[2], HumanMessage)
    assert stream_config.extract_final_content({"results": {"chat": {"content": "ok"}}}) == ""
    assert stream_config.should_stream_token("chat_agent", {"routing": {}, "plan": []}) is True
    assert stream_config.should_stream_token("router", {"routing": {}, "plan": []}) is False
    assert stream_config.map_exception(RuntimeError("boom")) == "服务暂时不可用，请稍后重试。"
    assert stream_config.initial_emitted_events == ()
    assert scheduled_calls == []
    assert call_order == ["add_user", "load_history"]
    assert saved_messages == [
        {
            "conversation_id": "507f1f77bcf86cd799439011",
            "role": "user",
            "content": "代理测试",
        }
    ]

    assert stream_config.on_answer_completed is not None
    asyncio.run(
        stream_config.on_answer_completed(
            "AI最终回复",
            {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "intermediate_tokens": 10,
                "total_tokens": 130,
            },
        )
    )
    assert merge_usage_calls == [
        {
            "stream_token_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "intermediate_tokens": 10,
                "total_tokens": 130,
            },
            "prompt_text": "代理测试",
            "completion_text": "AI最终回复",
        }
    ]
    assert saved_messages[-1] == {
        "conversation_id": "507f1f77bcf86cd799439011",
        "role": "assistant",
        "content": "AI最终回复",
        "token_usage": {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "intermediate_tokens": 3,
            "total_tokens": 21,
            "breakdown": [
                {
                    "node_name": "chat_agent",
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                }
            ],
        },
    }


def test_assistant_chat_new_conversation_injects_created_session_event(monkeypatch):
    captured: dict = {}
    scheduled_calls: list[dict] = []
    saved_messages: list[dict] = []
    merge_usage_calls: list[dict] = []
    load_history_called = False

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
        lambda **kwargs: scheduled_calls.append(kwargs),
    )
    monkeypatch.setattr(
        service_module,
        "add_message",
        lambda **kwargs: saved_messages.append(kwargs) or "507f1f77bcf86cd799439012",
    )
    monkeypatch.setattr(
        service_module,
        "merge_assistant_token_usage",
        lambda **kwargs: (
                merge_usage_calls.append(kwargs),
                {
                    "prompt_tokens": 6,
                    "completion_tokens": 4,
                    "intermediate_tokens": 0,
                    "total_tokens": 10,
                    "breakdown": None,
                },
        )[-1],
    )
    def _fake_load_history(**_kwargs):
        nonlocal load_history_called
        load_history_called = True
        return []

    monkeypatch.setattr(service_module, "load_history", _fake_load_history)

    response = service_module.assistant_chat(question="新建会话")

    assert isinstance(response, StreamingResponse)
    assert captured["question"] == "新建会话"
    stream_config = captured["config"]
    assert len(stream_config.initial_emitted_events) == 1
    session_event = stream_config.initial_emitted_events[0]
    assert session_event.type == MessageType.STATUS
    assert session_event.content.node == "conversation"
    assert session_event.content.state == "created"
    assert session_event.content.message == "会话创建成功"
    assert session_event.meta == {
        "conversation_uuid": "new-conv-uuid",
    }
    assert scheduled_calls == [
        {
            "conversation_uuid": "new-conv-uuid",
            "question": "新建会话",
        }
    ]
    assert load_history_called is False
    assert saved_messages == [
        {
            "conversation_id": "db-new-conv-uuid-100",
            "role": "user",
            "content": "新建会话",
        }
    ]
    assert stream_config.build_initial_state("x")["history_messages"] == []

    assert stream_config.on_answer_completed is not None
    asyncio.run(stream_config.on_answer_completed("AI结束回复"))
    assert merge_usage_calls == [
        {
            "stream_token_usage": None,
            "prompt_text": "新建会话",
            "completion_text": "AI结束回复",
        }
    ]
    assert saved_messages[-1] == {
        "conversation_id": "db-new-conv-uuid-100",
        "role": "assistant",
        "content": "AI结束回复",
        "token_usage": {
            "prompt_tokens": 6,
            "completion_tokens": 4,
            "intermediate_tokens": 0,
            "total_tokens": 10,
            "breakdown": None,
        },
    }


def test_load_history_reads_latest_window_and_returns_chronological(monkeypatch):
    captured: dict = {}

    def _fake_get_history(*, conversation_id: str, limit: int, ascending: bool):
        captured["conversation_id"] = conversation_id
        captured["limit"] = limit
        captured["ascending"] = ascending
        # 模拟数据库倒序（新 -> 旧）结果。
        return [
            HumanMessage(content="Q2"),
            AIMessage(content="A1"),
            HumanMessage(content="Q1"),
        ]

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
