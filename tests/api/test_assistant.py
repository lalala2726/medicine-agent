import json

from fastapi.testclient import TestClient

from app.api.routes import admin_assistant as assistant_module
from app.core.assistant_status import emit_status
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


def _extract_payloads(response_text: str) -> list[dict]:
    lines = [line for line in response_text.splitlines() if line.startswith("data: ")]
    return [json.loads(line[len("data: "):]) for line in lines]


def _answer_text(payload: dict) -> str:
    return payload["content"]["text"]


def test_assistant_streaming_response_from_workflow_chat_result(monkeypatch):
    monkeypatch.setattr(
        assistant_module,
        "ADMIN_WORKFLOW",
        _DummyGraph(final_state={"results": {"chat": {"content": "hello from workflow"}}}),
    )
    client = TestClient(app)

    response = client.post("/api/admin/assistant/chat", json={"question": "hi"})

    assert response.status_code == 200
    payloads = _extract_payloads(response.text)
    assert len(payloads) == 2
    assert _answer_text(payloads[0]) == "hello from workflow"
    assert payloads[0]["is_end"] is False
    assert _answer_text(payloads[1]) == ""
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
    payloads = _extract_payloads(response.text)
    assert _answer_text(payloads[0]) == "order done"
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
    payloads = _extract_payloads(response.text)
    assert _answer_text(payloads[0]) == "order done"
    assert payloads[0]["is_end"] is False
    assert _answer_text(payloads[1]) == ""
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
    payloads = _extract_payloads(response.text)
    assert _answer_text(payloads[0]) == "订"
    assert _answer_text(payloads[1]) == "单"
    assert _answer_text(payloads[2]) == ""
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
    payloads = _extract_payloads(response.text)
    assert _answer_text(payloads[0]) == "final answer"
    assert payloads[0]["is_end"] is False
    assert _answer_text(payloads[1]) == ""
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
    payloads = _extract_payloads(response.text)
    assert _answer_text(payloads[0]) == "final answer"
    assert payloads[0]["is_end"] is False
    assert _answer_text(payloads[1]) == ""
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
    payloads = _extract_payloads(response.text)
    assert _answer_text(payloads[0]) == "你"
    assert _answer_text(payloads[1]) == "好"
    assert _answer_text(payloads[2]) == ""
    assert payloads[2]["is_end"] is True


def test_assistant_streaming_response_from_astream_summary_tokens(monkeypatch):
    graph = _DummyAsyncGraph(
        events=[
            (
                "values",
                {
                    "routing": {
                        "route_target": "coordinator_agent",
                        "next_nodes": ["summary_agent"],
                        "is_final_stage": True,
                    },
                    "plan": [{"node_name": "summary_agent"}],
                },
            ),
            (
                "messages",
                (_DummyMessageChunk("总"), {"langgraph_node": "summary_agent"}),
            ),
            (
                "messages",
                (_DummyMessageChunk("结"), {"langgraph_node": "summary_agent"}),
            ),
            (
                "values",
                {"results": {"summary": {"content": "总结"}}},
            ),
        ]
    )
    monkeypatch.setattr(assistant_module, "ADMIN_WORKFLOW", graph)
    client = TestClient(app)

    response = client.post("/api/admin/assistant/chat", json={"question": "汇总一下"})

    assert response.status_code == 200
    payloads = _extract_payloads(response.text)
    assert _answer_text(payloads[0]) == "总"
    assert _answer_text(payloads[1]) == "结"
    assert _answer_text(payloads[2]) == ""
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
    payloads = _extract_payloads(response.text)
    assert _answer_text(payloads[0]) == "服务暂时不可用，请稍后重试。"
    assert payloads[1]["is_end"] is True


def test_assistant_streaming_status_events(monkeypatch):
    class _StatusAsyncGraph:
        async def astream(self, _state: dict, **_kwargs):
            emit_status(node="router", state="start", message="正在分析问题")
            yield ("values", {"routing": {"route_target": "order_agent"}, "plan": []})
            emit_status(node="router", state="end")

            emit_status(node="order", state="start", message="正在处理订单问题")
            emit_status(node="tool:get_order_list", state="start", message="正在查询订单信息")
            emit_status(node="tool:get_order_list", state="end")
            emit_status(node="order", state="end")

            yield ("messages", (_DummyMessageChunk("订"), {"langgraph_node": "order_agent"}))
            yield ("values", {"order_context": {"result": {"content": "订单"}}})

    monkeypatch.setattr(assistant_module, "ADMIN_WORKFLOW", _StatusAsyncGraph())
    client = TestClient(app)

    response = client.post("/api/admin/assistant/chat", json={"question": "查订单"})

    assert response.status_code == 200
    payloads = _extract_payloads(response.text)

    status_payloads = [item for item in payloads if item["type"] == "status"]
    status_contents = [item["content"] for item in status_payloads]
    assert {"node": "router", "state": "start", "message": "正在分析问题"} in status_contents
    assert {"node": "router", "state": "end"} in status_contents
    assert {"node": "order", "state": "start", "message": "正在处理订单问题"} in status_contents
    assert {"node": "order", "state": "end"} in status_contents
    assert {"node": "tool:get_order_list", "state": "start", "message": "正在查询订单信息"} in status_contents
    assert {"node": "tool:get_order_list", "state": "end"} in status_contents

    first_answer_index = next(
        index
        for index, payload in enumerate(payloads)
        if payload["type"] == "answer" and _answer_text(payload) == "订"
    )
    router_start_index = next(
        index
        for index, payload in enumerate(payloads)
        if payload["type"] == "status"
        and payload["content"].get("node") == "router"
        and payload["content"].get("state") == "start"
    )
    assert router_start_index < first_answer_index


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
