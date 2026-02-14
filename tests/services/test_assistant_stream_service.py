import asyncio
import json
from typing import Any, Callable

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.assistant_status import emit_function_call, emit_status
from app.schemas.sse_response import MessageType
from app.services.assistant_stream_service import (
    AssistantStreamConfig,
    StreamRuntimeState,
    create_streaming_response,
    drain_pending_events,
)


class _DummyMessageChunk:
    def __init__(self, content: str):
        self.content = content


class _DummyAsyncGraph:
    def __init__(self, events: list[tuple[str, object]]):
        self.events = events

    async def astream(self, _state: dict, **_kwargs):
        for event in self.events:
            yield event


class _DummySyncGraph:
    def __init__(self, final_state: dict | None = None, should_raise: bool = False):
        self.final_state = final_state or {}
        self.should_raise = should_raise

    def invoke(self, _state: dict) -> dict:
        if self.should_raise:
            raise RuntimeError("boom")
        return self.final_state


def _extract_payloads(response_text: str) -> list[dict]:
    lines = [line for line in response_text.splitlines() if line.startswith("data: ")]
    return [json.loads(line[len("data: "):]) for line in lines]


def _decode_sse_payload(raw_sse: str) -> dict:
    line = next(item for item in raw_sse.splitlines() if item.startswith("data: "))
    return json.loads(line[len("data: "):])


def _stream_with_config(config: AssistantStreamConfig) -> list[dict]:
    app = FastAPI()

    @app.get("/stream")
    async def stream():
        return create_streaming_response("question", config)

    client = TestClient(app)
    response = client.get("/stream")
    assert response.status_code == 200
    return _extract_payloads(response.text)


def _build_config(
        workflow: Any,
        *,
        should_stream_token: Callable[[str | None, dict[str, Any]], bool] | None = None,
        extract_final_content: Callable[[dict[str, Any]], str] | None = None,
        map_exception: Callable[[Exception], str] | None = None,
) -> AssistantStreamConfig:
    return AssistantStreamConfig(
        workflow=workflow,
        build_initial_state=lambda question: {
            "question": question,
            "routing": {},
            "plan": [],
            "final": "fallback",
        },
        extract_final_content=extract_final_content or (lambda state: state.get("final", "")),
        should_stream_token=should_stream_token or (lambda stream_node, _state: stream_node == "chat_agent"),
        build_stream_config=lambda: {"trace_id": "demo"},
        invoke_sync=lambda state: workflow.invoke(state),
        map_exception=map_exception or (lambda exc: f"处理失败: {exc}"),
    )


def test_stream_service_hides_node_for_function_call_but_keeps_status_node():
    class _StatusAsyncGraph:
        async def astream(self, _state: dict, **_kwargs):
            emit_status(node="router", state="start", message="开始")
            emit_function_call(node="tool:get_order_list", state="start", message="调用中")
            emit_function_call(node="tool:get_order_list", state="end")
            emit_status(node="router", state="end")
            yield ("values", {"final": "done"})

    payloads = _stream_with_config(_build_config(_StatusAsyncGraph(), should_stream_token=lambda *_: False))

    status_payloads = [item for item in payloads if item["type"] == "status"]
    function_call_payloads = [item for item in payloads if item["type"] == "function_call"]
    answer_payloads = [item for item in payloads if item["type"] == "answer"]

    assert {"node": "router", "state": "start", "message": "开始"} in [
        item["content"] for item in status_payloads
    ]
    assert {"node": "router", "state": "end"} in [item["content"] for item in status_payloads]
    assert {"state": "start", "message": "调用中"} in [
        item["content"] for item in function_call_payloads
    ]
    assert {"state": "end"} in [item["content"] for item in function_call_payloads]
    assert all("node" not in item["content"] for item in function_call_payloads)

    assert any(item["content"].get("text") == "done" for item in answer_payloads)
    assert answer_payloads[-1]["is_end"] is True
    assert answer_payloads[-1]["content"]["text"] == ""


def test_stream_service_streams_message_tokens_and_skips_fallback():
    graph = _DummyAsyncGraph(
        events=[
            ("values", {"routing": {}, "plan": []}),
            ("messages", (_DummyMessageChunk("你"), {"langgraph_node": "chat_agent"})),
            ("messages", (_DummyMessageChunk("好"), {"langgraph_node": "chat_agent"})),
            ("values", {"final": "不会走兜底"}),
        ]
    )
    payloads = _stream_with_config(_build_config(graph))

    answer_payloads = [item for item in payloads if item["type"] == "answer"]
    non_end_texts = [item["content"]["text"] for item in answer_payloads if not item["is_end"]]
    assert non_end_texts == ["你", "好"]
    assert answer_payloads[-1]["is_end"] is True
    assert answer_payloads[-1]["content"]["text"] == ""


def test_stream_service_uses_fallback_when_no_tokens():
    graph = _DummyAsyncGraph(
        events=[
            ("messages", (_DummyMessageChunk("忽略"), {"langgraph_node": "other_agent"})),
            ("values", {"final": "fallback text"}),
        ]
    )
    payloads = _stream_with_config(_build_config(graph))

    answer_payloads = [item for item in payloads if item["type"] == "answer"]
    non_end_texts = [item["content"]["text"] for item in answer_payloads if not item["is_end"]]
    assert non_end_texts == ["fallback text"]
    assert answer_payloads[-1]["is_end"] is True


def test_stream_service_maps_exception_to_answer():
    class _ErrorAsyncGraph:
        async def astream(self, _state: dict, **_kwargs):
            raise RuntimeError("爆炸")
            yield  # pragma: no cover

    payloads = _stream_with_config(
        _build_config(
            _ErrorAsyncGraph(),
            map_exception=lambda exc: f"错误映射: {exc}",
        )
    )

    answer_payloads = [item for item in payloads if item["type"] == "answer"]
    non_end_texts = [item["content"]["text"] for item in answer_payloads if not item["is_end"]]
    assert non_end_texts == ["错误映射: 爆炸"]
    assert answer_payloads[-1]["is_end"] is True


def test_stream_service_invoke_fallback_path():
    graph = _DummySyncGraph(final_state={"final": "sync fallback"})
    payloads = _stream_with_config(_build_config(graph))

    answer_payloads = [item for item in payloads if item["type"] == "answer"]
    non_end_texts = [item["content"]["text"] for item in answer_payloads if not item["is_end"]]
    assert non_end_texts == ["sync fallback"]
    assert answer_payloads[-1]["is_end"] is True


def test_drain_pending_events_consumes_tail_events():
    async def _run():
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        await queue.put(
            (
                "emitted",
                {
                    "type": "status",
                    "content": {"node": "router", "state": "start", "message": "准备中"},
                },
            )
        )
        await queue.put(
            (
                "graph",
                ("messages", (_DummyMessageChunk("T"), {"langgraph_node": "chat_agent"})),
            )
        )
        await queue.put(("graph", ("values", {"final": "tail"})))
        await queue.put(("error", RuntimeError("tail error")))
        runtime_state = StreamRuntimeState(latest_state={"final": "init"})
        drain_result = await drain_pending_events(
            queue=queue,
            runtime_state=runtime_state,
            should_stream_token=lambda stream_node, _state: stream_node == "chat_agent",
            hide_node_types={MessageType.FUNCTION_CALL},
            map_exception=lambda exc: f"错误: {exc}",
        )
        return drain_result, runtime_state

    drain_result, runtime_state = asyncio.run(_run())
    rendered_events = drain_result.rendered_events
    decoded = [_decode_sse_payload(item) for item in rendered_events]

    assert {"node": "router", "state": "start", "message": "准备中"} in [
        item["content"] for item in decoded if item["type"] == "status"
    ]
    assert "T" in [
        item["content"]["text"]
        for item in decoded
        if item["type"] == "answer" and not item["is_end"]
    ]
    assert "错误: tail error" in [
        item["content"]["text"]
        for item in decoded
        if item["type"] == "answer" and not item["is_end"]
    ]
    assert runtime_state.latest_state["final"] == "tail"
    assert runtime_state.has_streamed_output is True
    assert runtime_state.has_emitted_error is True
