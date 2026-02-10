import json

from fastapi.testclient import TestClient

from app.api.routes import admin_assistant as assistant_module
from app.main import app


class _DummyGraph:
    def __init__(self, final_state: dict | None = None, should_raise: bool = False) -> None:
        self.final_state = final_state or {}
        self.should_raise = should_raise
        self.captured_config = None

    def invoke(self, _state: dict, config: dict | None = None) -> dict:
        self.captured_config = config
        if self.should_raise:
            raise RuntimeError("graph failed")
        return self.final_state


class _DummyMessageChunk:
    def __init__(self, content: str):
        self.content = content


class _DummyAsyncGraph:
    def __init__(self, events: list[tuple[str, object]]):
        self.events = events

    async def astream(self, _state: dict, **_kwargs):
        for event in self.events:
            yield event


def test_assistant_streaming_response_from_workflow_chat_result(monkeypatch):
    monkeypatch.setattr(
        assistant_module,
        "ADMIN_WORKFLOW",
        _DummyGraph(final_state={"results": {"chat": {"content": "hello from workflow"}}}),
    )
    client = TestClient(app)

    response = client.post("/api/admin/assistant/chat", json={"question": "hi"})

    assert response.status_code == 200
    lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    assert len(lines) == 2
    payloads = [json.loads(line[len("data: "):]) for line in lines]
    assert payloads[0]["content"] == "hello from workflow"
    assert payloads[0]["is_end"] is False
    assert payloads[1]["content"] == ""
    assert payloads[1]["is_end"] is True


def test_assistant_streaming_response_from_order_context(monkeypatch):
    monkeypatch.setattr(
        assistant_module,
        "ADMIN_WORKFLOW",
        _DummyGraph(final_state={"order_context": {"result": {"content": "order done"}}}),
    )
    client = TestClient(app)

    response = client.post("/api/admin/assistant/chat", json={"question": "查订单"})

    assert response.status_code == 200
    lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    payloads = [json.loads(line[len("data: "):]) for line in lines]
    assert payloads[0]["content"] == "order done"
    assert payloads[1]["is_end"] is True


def test_assistant_streaming_response_from_order_context_chunks_with_invoke_fallback(monkeypatch):
    monkeypatch.setattr(
        assistant_module,
        "ADMIN_WORKFLOW",
        _DummyGraph(
            final_state={
                "order_context": {
                    "result": {"content": "order done", "is_end": True},
                    "stream_chunks": ["order ", "done"],
                }
            }
        ),
    )
    client = TestClient(app)

    response = client.post("/api/admin/assistant/chat", json={"question": "查订单"})

    assert response.status_code == 200
    lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    payloads = [json.loads(line[len("data: "):]) for line in lines]
    assert payloads[0]["content"] == "order done"
    assert payloads[0]["is_end"] is False
    assert payloads[1]["content"] == ""
    assert payloads[1]["is_end"] is True


def test_assistant_streaming_response_from_astream_order_tokens(monkeypatch):
    graph = _DummyAsyncGraph(
        events=[
            (
                "values",
                {"routing": {"route_target": "order_agent"}, "plan": []},
            ),
            (
                "messages",
                (_DummyMessageChunk("订"), {"langgraph_node": "order_agent"}),
            ),
            (
                "messages",
                (_DummyMessageChunk("单"), {"langgraph_node": "order_agent"}),
            ),
            (
                "values",
                {"order_context": {"result": {"content": "订单"}}},
            ),
        ]
    )
    monkeypatch.setattr(assistant_module, "ADMIN_WORKFLOW", graph)
    client = TestClient(app)

    response = client.post("/api/admin/assistant/chat", json={"question": "查订单"})

    assert response.status_code == 200
    lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    payloads = [json.loads(line[len("data: "):]) for line in lines]
    assert payloads[0]["content"] == "订"
    assert payloads[1]["content"] == "单"
    assert payloads[2]["content"] == ""
    assert payloads[2]["is_end"] is True


def test_assistant_streaming_response_from_astream_non_order_tokens(monkeypatch):
    graph = _DummyAsyncGraph(
        events=[
            (
                "messages",
                (_DummyMessageChunk("忽略"), {"langgraph_node": "coordinator_agent"}),
            ),
            (
                "values",
                {"results": {"chat": {"content": "final answer"}}},
            ),
        ]
    )
    monkeypatch.setattr(assistant_module, "ADMIN_WORKFLOW", graph)
    client = TestClient(app)

    response = client.post("/api/admin/assistant/chat", json={"question": "hi"})

    assert response.status_code == 200
    lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    payloads = [json.loads(line[len("data: "):]) for line in lines]
    assert payloads[0]["content"] == "final answer"
    assert payloads[0]["is_end"] is False
    assert payloads[1]["content"] == ""
    assert payloads[1]["is_end"] is True


def test_assistant_does_not_stream_order_tokens_when_order_is_not_final(monkeypatch):
    graph = _DummyAsyncGraph(
        events=[
            (
                "values",
                {
                    "routing": {
                        "route_target": "coordinator_agent",
                        "next_nodes": ["order_agent"],
                        "is_final_stage": False,
                    },
                    "plan": [{"node_name": "order_agent"}],
                },
            ),
            (
                "messages",
                (_DummyMessageChunk("中间结果"), {"langgraph_node": "order_agent"}),
            ),
            (
                "values",
                {"results": {"chat": {"content": "final answer"}}},
            ),
        ]
    )
    monkeypatch.setattr(assistant_module, "ADMIN_WORKFLOW", graph)
    client = TestClient(app)

    response = client.post("/api/admin/assistant/chat", json={"question": "hi"})

    assert response.status_code == 200
    lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    payloads = [json.loads(line[len("data: "):]) for line in lines]
    assert payloads[0]["content"] == "final answer"
    assert payloads[0]["is_end"] is False
    assert payloads[1]["content"] == ""
    assert payloads[1]["is_end"] is True


def test_assistant_streaming_response_from_astream_chat_tokens(monkeypatch):
    graph = _DummyAsyncGraph(
        events=[
            (
                "messages",
                (_DummyMessageChunk("你"), {"langgraph_node": "chat_agent"}),
            ),
            (
                "messages",
                (_DummyMessageChunk("好"), {"langgraph_node": "chat_agent"}),
            ),
            (
                "values",
                {"results": {"chat": {"content": "你好"}}},
            ),
        ]
    )
    monkeypatch.setattr(assistant_module, "ADMIN_WORKFLOW", graph)
    client = TestClient(app)

    response = client.post("/api/admin/assistant/chat", json={"question": "你好"})

    assert response.status_code == 200
    lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    payloads = [json.loads(line[len("data: "):]) for line in lines]
    assert payloads[0]["content"] == "你"
    assert payloads[1]["content"] == "好"
    assert payloads[2]["content"] == ""
    assert payloads[2]["is_end"] is True


def test_assistant_streaming_response_when_workflow_raises(monkeypatch):
    monkeypatch.setattr(
        assistant_module,
        "ADMIN_WORKFLOW",
        _DummyGraph(should_raise=True),
    )
    client = TestClient(app)

    response = client.post("/api/admin/assistant/chat", json={"question": "hi"})

    assert response.status_code == 200
    lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    payloads = [json.loads(line[len("data: "):]) for line in lines]
    assert payloads[0]["content"] == "服务暂时不可用，请稍后重试。"
    assert payloads[1]["is_end"] is True


def test_assistant_requires_question():
    client = TestClient(app)

    response = client.post("/api/admin/assistant/chat", json={"question": ""})

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == 400
    assert body["message"] == "问题不能为空"


def test_invoke_admin_workflow_passes_langsmith_config(monkeypatch):
    graph = _DummyGraph(final_state={"results": {"chat": {"content": "ok"}}})
    monkeypatch.setattr(assistant_module, "ADMIN_WORKFLOW", graph)
    monkeypatch.setattr(
        assistant_module,
        "build_langsmith_runnable_config",
        lambda **kwargs: {
            "run_name": "admin_assistant_graph",
            "tags": ["admin-assistant", "langgraph"],
            "metadata": {"entrypoint": "api.admin_assistant.chat"},
        },
    )

    state = {"user_input": "hello"}
    result = assistant_module._invoke_admin_workflow(state)

    assert result["results"]["chat"]["content"] == "ok"
    assert graph.captured_config is not None
    assert graph.captured_config["run_name"] == "admin_assistant_graph"
