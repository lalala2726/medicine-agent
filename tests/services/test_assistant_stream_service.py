import asyncio
import json
from typing import Any, Callable

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.assistant_status import emit_function_call, emit_sse_response, emit_status
from app.schemas.sse_response import AssistantResponse, Content, MessageType
from app.services.assistant_stream_service import (
    AssistantStreamConfig,
    StreamRuntimeState,
    create_streaming_response,
    drain_pending_events,
)


class _DummyMessageChunk:
    def __init__(
            self,
            content: str,
            *,
            chunk_id: str | None = None,
            usage_metadata: dict[str, int] | None = None,
            response_metadata: dict[str, Any] | None = None,
    ):
        self.content = content
        self.id = chunk_id
        self.usage_metadata = usage_metadata
        self.response_metadata = response_metadata


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
        initial_emitted_events: tuple[AssistantResponse, ...] = (),
        on_answer_completed: Callable[..., None] | None = None,
) -> AssistantStreamConfig:
    return AssistantStreamConfig(
        workflow=workflow,
        build_initial_state=lambda question: {
            "question": question,
            "routing": {},
            "plan": [],
            "final": "fallback",
            "execution_traces": [],
        },
        extract_final_content=extract_final_content or (lambda state: state.get("final", "")),
        should_stream_token=should_stream_token or (lambda stream_node, _state: stream_node == "chat_agent"),
        build_stream_config=lambda: {"trace_id": "demo"},
        invoke_sync=lambda state: workflow.invoke(state),
        map_exception=map_exception or (lambda exc: f"处理失败: {exc}"),
        on_answer_completed=on_answer_completed,
        initial_emitted_events=initial_emitted_events,
    )


def test_stream_service_hides_node_for_function_call_but_keeps_status_node():
    class _StatusAsyncGraph:
        async def astream(self, _state: dict, **_kwargs):
            emit_status(node="router", state="start", message="开始")
            emit_function_call(
                node="tool:get_order_list",
                parent_node="order",
                state="start",
                message="调用中",
            )
            emit_function_call(
                node="tool:get_order_list",
                parent_node="order",
                state="end",
            )
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
    assert {"parent_node": "order", "state": "start", "message": "调用中"} in [
        item["content"] for item in function_call_payloads
    ]
    assert {"parent_node": "order", "state": "end"} in [
        item["content"] for item in function_call_payloads
    ]
    assert all("node" not in item["content"] for item in function_call_payloads)

    assert any(item["content"].get("text") == "done" for item in answer_payloads)
    assert answer_payloads[-1]["is_end"] is True
    assert answer_payloads[-1]["content"]["text"] == ""


def test_stream_service_emits_initial_session_event_before_graph_output():
    graph = _DummyAsyncGraph(events=[("values", {"final": ""})])
    session_event = AssistantResponse(
        content=Content(node="conversation", state="created", message="会话创建成功"),
        type=MessageType.STATUS,
        meta={"conversation_uuid": "conv-1", "conversation_id": "mongo-1"},
    )
    payloads = _stream_with_config(
        _build_config(
            graph,
            should_stream_token=lambda *_: False,
            extract_final_content=lambda _state: "",
            initial_emitted_events=(session_event,),
        )
    )

    # 首条事件应是预注入的会话创建通知。
    assert payloads[0]["type"] == "status"
    assert payloads[0]["content"] == {
        "node": "conversation",
        "state": "created",
        "message": "会话创建成功",
    }
    assert payloads[0]["meta"] == {
        "conversation_uuid": "conv-1",
        "conversation_id": "mongo-1",
    }
    assert payloads[-1]["is_end"] is True


def test_stream_service_accepts_custom_assistant_response_answer_and_skips_fallback():
    class _CustomAnswerGraph:
        async def astream(self, _state: dict, **_kwargs):
            emit_sse_response(
                AssistantResponse(
                    content=Content(text="manual answer", message="附加参数"),
                    type=MessageType.ANSWER,
                    meta={"conversation_uuid": "conv-2"},
                    is_end=True,
                )
            )
            yield ("values", {"final": "fallback should be skipped"})

    payloads = _stream_with_config(
        _build_config(
            _CustomAnswerGraph(),
            should_stream_token=lambda *_: False,
        )
    )

    answer_payloads = [item for item in payloads if item["type"] == "answer"]
    non_end_payloads = [item for item in answer_payloads if not item["is_end"]]
    assert len(non_end_payloads) == 1
    assert non_end_payloads[0]["content"]["text"] == "manual answer"
    assert non_end_payloads[0]["content"]["message"] == "附加参数"
    assert non_end_payloads[0]["meta"] == {"conversation_uuid": "conv-2"}
    assert non_end_payloads[0]["is_end"] is False

    # 结束包仍由流式引擎统一输出，避免业务侧提前关闭流。
    assert answer_payloads[-1]["is_end"] is True
    assert answer_payloads[-1]["content"]["text"] == ""


def test_stream_service_hides_node_for_custom_function_call_response():
    class _CustomFunctionCallGraph:
        async def astream(self, _state: dict, **_kwargs):
            emit_sse_response(
                AssistantResponse(
                    content=Content(
                        node="tool:get_order_list",
                        parent_node="order",
                        state="start",
                        message="调用中",
                    ),
                    type=MessageType.FUNCTION_CALL,
                )
            )
            yield ("values", {"final": ""})

    payloads = _stream_with_config(
        _build_config(
            _CustomFunctionCallGraph(),
            should_stream_token=lambda *_: False,
            extract_final_content=lambda _state: "",
        )
    )

    function_call_payloads = [item for item in payloads if item["type"] == "function_call"]
    assert len(function_call_payloads) == 1
    assert function_call_payloads[0]["content"] == {
        "parent_node": "order",
        "state": "start",
        "message": "调用中",
    }

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


def test_stream_service_calls_answer_completed_callback_with_aggregated_text():
    """测试目标：流结束回调输出完整 answer；成功标准：回调收到拼接后的文本。"""

    graph = _DummyAsyncGraph(
        events=[
            ("messages", (_DummyMessageChunk("你"), {"langgraph_node": "chat_agent"})),
            ("messages", (_DummyMessageChunk("好"), {"langgraph_node": "chat_agent"})),
            ("values", {"final": "不会走兜底"}),
        ]
    )
    completed_payloads: list[dict[str, Any]] = []
    payloads = _stream_with_config(
        _build_config(
            graph,
            on_answer_completed=lambda text, _usage, trace: completed_payloads.append(
                {"text": text, "trace": trace}
            ),
        )
    )

    answer_payloads = [item for item in payloads if item["type"] == "answer"]
    non_end_texts = [item["content"]["text"] for item in answer_payloads if not item["is_end"]]
    assert non_end_texts == ["你", "好"]
    assert completed_payloads[0]["text"] == "你好"
    assert completed_payloads[0]["trace"] is None


def test_stream_service_collects_token_usage_summary_for_callback():
    """测试目标：token 汇总按全流程聚合；成功标准：unknown 模型名可由 execution_trace 补齐。"""

    graph = _DummyAsyncGraph(
        events=[
            ("values", {"routing": {}, "plan": []}),
            (
                "messages",
                (
                    _DummyMessageChunk(
                        "",
                        chunk_id="order-1",
                        usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
                    ),
                    {"langgraph_node": "order_agent"},
                ),
            ),
            (
                "messages",
                (
                    _DummyMessageChunk(
                        "",
                        chunk_id="order-1",
                        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                    ),
                    {"langgraph_node": "order_agent"},
                ),
            ),
            (
                "messages",
                (
                    _DummyMessageChunk(
                        "你",
                        chunk_id="chat-1",
                        usage_metadata={"input_tokens": 20, "output_tokens": 4, "total_tokens": 24},
                        response_metadata={"model_name": "qwen-plus"},
                    ),
                    {"langgraph_node": "chat_agent"},
                ),
            ),
            (
                "messages",
                (
                    _DummyMessageChunk(
                        "好",
                        chunk_id="chat-1",
                        usage_metadata={"input_tokens": 20, "output_tokens": 6, "total_tokens": 26},
                        response_metadata={"model_name": "qwen-plus"},
                    ),
                    {"langgraph_node": "chat_agent"},
                ),
            ),
            (
                "values",
                {
                    "final": "",
                    "execution_traces": [
                        {
                            "node_name": "order_agent",
                            "model_name": "qwen3-max",
                            "input_messages": [],
                            "output_text": "订单节点输出",
                            "tool_calls": [],
                        },
                        {
                            "node_name": "chat_agent",
                            "model_name": "qwen-plus",
                            "input_messages": [],
                            "output_text": "聊天节点输出",
                            "tool_calls": [],
                        },
                    ],
                },
            ),
        ]
    )
    callback_payloads: list[dict[str, Any]] = []

    def _callback(
            answer_text: str,
            token_usage: dict[str, Any] | None,
            execution_trace: list[dict[str, Any]] | None,
    ) -> None:
        callback_payloads.append(
            {
                "answer_text": answer_text,
                "token_usage": token_usage,
                "execution_trace": execution_trace,
            }
        )

    payloads = _stream_with_config(
        _build_config(
            graph,
            on_answer_completed=_callback,
        )
    )

    answer_payloads = [item for item in payloads if item["type"] == "answer"]
    non_end_texts = [item["content"]["text"] for item in answer_payloads if not item["is_end"]]
    assert non_end_texts == ["你", "好"]

    assert len(callback_payloads) == 1
    assert callback_payloads[0]["answer_text"] == "你好"
    usage = callback_payloads[0]["token_usage"]
    assert usage is not None
    assert usage["prompt_tokens"] == 30
    assert usage["completion_tokens"] == 11
    assert usage["total_tokens"] == 41

    breakdown_by_node_model = {
        f"{item['node_name']}::{item['model_name']}": item for item in usage["breakdown"]
    }
    assert breakdown_by_node_model["order_agent::qwen3-max"] == {
        "node_name": "order_agent",
        "model_name": "qwen3-max",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
    assert breakdown_by_node_model["chat_agent::qwen-plus"] == {
        "node_name": "chat_agent",
        "model_name": "qwen-plus",
        "prompt_tokens": 20,
        "completion_tokens": 6,
        "total_tokens": 26,
    }
    assert callback_payloads[0]["execution_trace"] is not None


def test_on_answer_completed_receives_three_args():
    """测试目标：回调参数统一三参；成功标准：回调可拿到 answer/token_usage/execution_trace。"""

    graph = _DummyAsyncGraph(
        events=[
            ("messages", (_DummyMessageChunk("好"), {"langgraph_node": "chat_agent"})),
            (
                "values",
                {
                    "final": "",
                    "execution_traces": [
                        {
                            "node_name": "chat_agent",
                            "model_name": "qwen-flash",
                            "input_messages": [{"role": "human", "content": "hello"}],
                            "output_text": "好",
                            "tool_calls": [],
                        }
                    ],
                },
            ),
        ]
    )

    callback_args: list[dict[str, Any]] = []

    def _callback(
            answer_text: str,
            token_usage: dict[str, Any] | None,
            execution_trace: list[dict[str, Any]] | None,
    ) -> None:
        callback_args.append(
            {
                "answer_text": answer_text,
                "token_usage": token_usage,
                "execution_trace": execution_trace,
            }
        )

    _stream_with_config(_build_config(graph, on_answer_completed=_callback))

    assert len(callback_args) == 1
    assert callback_args[0]["answer_text"] == "好"
    assert isinstance(callback_args[0]["execution_trace"], list)


def test_on_answer_completed_supports_four_args_with_has_error():
    """测试目标：4 参回调可收到 has_error 标志。"""

    class _ErrorAsyncGraph:
        async def astream(self, _state: dict, **_kwargs):
            raise RuntimeError("爆炸")
            yield  # pragma: no cover

    callback_args: list[dict[str, Any]] = []

    def _callback(
            answer_text: str,
            token_usage: dict[str, Any] | None,
            execution_trace: list[dict[str, Any]] | None,
            has_error: bool,
    ) -> None:
        callback_args.append(
            {
                "answer_text": answer_text,
                "token_usage": token_usage,
                "execution_trace": execution_trace,
                "has_error": has_error,
            }
        )

    _stream_with_config(
        _build_config(
            _ErrorAsyncGraph(),
            map_exception=lambda exc: f"错误映射: {exc}",
            on_answer_completed=_callback,
        )
    )

    assert len(callback_args) == 1
    assert callback_args[0]["answer_text"] == "错误映射: 爆炸"
    assert callback_args[0]["has_error"] is True


def test_execution_trace_summary_reads_full_graph_nodes():
    """测试目标：execution_trace 汇总覆盖全图节点；成功标准：含非 LLM 节点并保持空输入/空工具。"""

    graph = _DummyAsyncGraph(
        events=[
            (
                "values",
                {
                    "final": "",
                    "execution_traces": [
                        {
                            "node_name": "gateway_router",
                            "model_name": "unknown",
                            "input_messages": [],
                            "output_text": "{\"routing\": {}}",
                            "tool_calls": [],
                        },
                        {
                            "node_name": "planner",
                            "model_name": "unknown",
                            "input_messages": [],
                            "output_text": "{\"routing\": {\"next_nodes\": []}}",
                            "tool_calls": [],
                        },
                    ],
                },
            ),
        ]
    )
    callback_payloads: list[list[dict[str, Any]] | None] = []

    def _callback(
            _answer_text: str,
            _token_usage: dict[str, Any] | None,
            execution_trace: list[dict[str, Any]] | None,
    ) -> None:
        callback_payloads.append(execution_trace)

    _stream_with_config(_build_config(graph, should_stream_token=lambda *_: False, on_answer_completed=_callback))

    assert callback_payloads
    trace = callback_payloads[0]
    assert trace is not None
    assert trace[0]["node_name"] == "gateway_router"
    assert trace[0]["input_messages"] == []
    assert trace[0]["tool_calls"] == []
    assert trace[1]["node_name"] == "planner"


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


def test_stream_service_skips_empty_fallback_text():
    graph = _DummyAsyncGraph(
        events=[
            ("values", {"final": ""}),
        ]
    )
    payloads = _stream_with_config(
        _build_config(
            graph,
            extract_final_content=lambda _state: "",
        )
    )

    answer_payloads = [item for item in payloads if item["type"] == "answer"]
    non_end_payloads = [item for item in answer_payloads if not item["is_end"]]
    assert non_end_payloads == []
    assert answer_payloads[-1]["is_end"] is True
    assert answer_payloads[-1]["content"]["text"] == ""


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
