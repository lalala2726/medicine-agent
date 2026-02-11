import json

from fastapi.testclient import TestClient

from app.api.routes import admin_assistant as assistant_module
from app.core.assistant_status import emit_function_call, emit_status
from app.main import app
from app.schemas.sse_response import MessageType
from app.services.assistant_stream_service import AssistantStreamConfig


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
            emit_function_call(node="tool:get_order_list", state="start", message="正在查询订单信息")
            emit_function_call(node="tool:get_order_list", state="end")
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

    function_call_payloads = [item for item in payloads if item["type"] == "function_call"]
    function_call_contents = [item["content"] for item in function_call_payloads]
    assert {"state": "start", "message": "正在查询订单信息"} in function_call_contents
    assert {"state": "end"} in function_call_contents
    assert all("node" not in item for item in function_call_contents)


def test_assistant_streaming_function_call_timely_events(monkeypatch):
    class _TimelyAsyncGraph:
        async def astream(self, _state: dict, **_kwargs):
            emit_function_call(node="tool:get_order_list", state="start", message="正在查询订单信息")
            emit_function_call(
                node="tool:get_order_list",
                state="timely",
                message="订单信息正在持续处理中",
            )
            yield ("values", {"results": {"chat": {"content": "处理中"}}})

    monkeypatch.setattr(assistant_module, "ADMIN_WORKFLOW", _TimelyAsyncGraph())
    client = TestClient(app)

    response = client.post("/api/admin/assistant/chat", json={"question": "查订单"})

    assert response.status_code == 200
    payloads = _extract_payloads(response.text)
    function_call_payloads = [item for item in payloads if item["type"] == "function_call"]
    function_call_contents = [item["content"] for item in function_call_payloads]

    assert {"state": "start", "message": "正在查询订单信息"} in function_call_contents
    assert {"state": "timely", "message": "订单信息正在持续处理中"} in function_call_contents
    assert all(item.get("state") != "end" for item in function_call_contents)
    assert all("node" not in item for item in function_call_contents)


def test_assistant_route_delegates_to_stream_service(monkeypatch):
    captured: dict = {}

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

        from fastapi.responses import StreamingResponse

        return StreamingResponse(_stream(), media_type="text/event-stream")

    monkeypatch.setattr(
        assistant_module,
        "create_streaming_response",
        _fake_create_streaming_response,
    )
    client = TestClient(app)

    response = client.post("/api/admin/assistant/chat", json={"question": "代理测试"})

    assert response.status_code == 200
    payloads = _extract_payloads(response.text)
    assert _answer_text(payloads[0]) == "delegated"
    assert payloads[1]["is_end"] is True

    stream_config = captured["config"]
    assert captured["question"] == "代理测试"
    assert isinstance(stream_config, AssistantStreamConfig)
    assert stream_config.workflow is assistant_module.ADMIN_WORKFLOW
    assert stream_config.build_initial_state("x")["user_input"] == "x"
    assert (
            stream_config.extract_final_content({"results": {"chat": {"content": "ok"}}})
            == "ok"
    )
    assert stream_config.should_stream_token("chat_agent", {"routing": {}, "plan": []}) is True
    assert stream_config.should_stream_token("router", {"routing": {}, "plan": []}) is False
    assert stream_config.map_exception(RuntimeError("boom")) == "服务暂时不可用，请稍后重试。"
    assert MessageType.FUNCTION_CALL in stream_config.hide_node_types


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
