import app.utils.streaming_utils as streaming_utils_module
from app.agent.admin.node.runtime_context import evaluate_failure_by_policy


class _DummyChunk:
    def __init__(self, content: str):
        self.content = content


class _StreamModel:
    def __init__(self):
        self.bind_tools_called = False
        self.stream_called = False

    def bind_tools(self, _tools):
        self.bind_tools_called = True
        return self

    def stream(self, _messages):
        self.stream_called = True
        yield _DummyChunk("hello ")
        yield _DummyChunk("world")


def test_invoke_with_optional_stream_prefers_stream_chunks():
    model = _StreamModel()

    content, chunks = streaming_utils_module.invoke_with_optional_stream(
        model,
        messages=["m1"],
        tools=[object()],
        enable_stream=True,
    )

    assert model.bind_tools_called is True
    assert model.stream_called is True
    assert chunks == ["hello ", "world"]
    assert content == "hello world"


def test_invoke_with_optional_stream_falls_back_when_stream_not_available(monkeypatch):
    class _NoStreamModel:
        def bind_tools(self, _tools):
            return self

    captured: dict = {}

    def fake_invoke(llm, messages, *, tools=None, max_tool_rounds=0):
        captured["llm"] = llm
        captured["messages"] = messages
        captured["tools"] = tools
        captured["max_tool_rounds"] = max_tool_rounds
        return "fallback"

    monkeypatch.setattr(streaming_utils_module, "invoke", fake_invoke)
    model = _NoStreamModel()

    content, chunks = streaming_utils_module.invoke_with_optional_stream(
        model,
        messages=["m2"],
        tools=[object()],
        enable_stream=True,
        max_tool_rounds=7,
    )

    assert content == "fallback"
    assert chunks == []
    assert captured["llm"] is model
    assert captured["messages"] == ["m2"]
    assert len(captured["tools"]) == 1
    assert captured["max_tool_rounds"] == 7


def test_invoke_with_optional_stream_falls_back_when_stream_returns_empty(monkeypatch):
    class _EmptyStreamModel:
        def stream(self, _messages):
            yield _DummyChunk("")

    monkeypatch.setattr(
        streaming_utils_module, "invoke", lambda *_args, **_kwargs: "fallback-empty"
    )
    model = _EmptyStreamModel()

    content, chunks = streaming_utils_module.invoke_with_optional_stream(
        model,
        messages=["m3"],
        enable_stream=True,
    )

    assert content == "fallback-empty"
    assert chunks == []


class _ToolResponse:
    def __init__(self, content: str, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class _PolicyModel:
    def __init__(self, responses):
        self._responses = list(responses)

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        return self._responses.pop(0)


class _FailingTool:
    name = "dummy_tool"

    def __init__(self):
        self.calls = 0

    async def ainvoke(self, _args):
        self.calls += 1
        return {"error": "tool failed"}


class _FlakyTool:
    name = "dummy_tool"

    def __init__(self):
        self.calls = 0

    async def ainvoke(self, _args):
        self.calls += 1
        if self.calls == 1:
            return {"error": "first fail"}
        return {"ok": True}


def test_invoke_with_policy_marks_threshold_hit_after_consecutive_tool_errors():
    tool_call = {"id": "tc1", "name": "dummy_tool", "args": {}}
    model = _PolicyModel(
        responses=[
            _ToolResponse("", tool_calls=[tool_call]),
            _ToolResponse("", tool_calls=[tool_call]),
        ]
    )
    content, diagnostics = streaming_utils_module.invoke_with_policy(
        model,
        messages=["m1"],
        tools=[_FailingTool()],
        error_marker_prefix="__ERROR__:",
        tool_error_counting="consecutive",
        max_tool_errors=2,
    )
    assert content.startswith("__ERROR__:")
    assert diagnostics["threshold_hit"] is True
    assert diagnostics["tool_calls"] == 2
    assert diagnostics["tool_errors_total"] == 2
    assert diagnostics["tool_errors_consecutive_peak"] == 2


def test_invoke_with_policy_resets_consecutive_count_after_tool_success():
    tool_call = {"id": "tc1", "name": "dummy_tool", "args": {}}
    model = _PolicyModel(
        responses=[
            _ToolResponse("", tool_calls=[tool_call]),
            _ToolResponse("", tool_calls=[tool_call]),
            _ToolResponse("final text", tool_calls=[]),
        ]
    )
    content, diagnostics = streaming_utils_module.invoke_with_policy(
        model,
        messages=["m1"],
        tools=[_FlakyTool()],
        error_marker_prefix="__ERROR__:",
        tool_error_counting="consecutive",
        max_tool_errors=2,
    )
    assert content == "final text"
    assert diagnostics["threshold_hit"] is False
    assert diagnostics["tool_errors_total"] == 1
    assert diagnostics["tool_errors_consecutive_peak"] == 1


def test_failure_policy_mode_behaviors():
    failed_by_marker = evaluate_failure_by_policy(
        "__ERROR__: bad data",
        diagnostics={"threshold_hit": False},
        policy={"mode": "marker_only", "error_marker_prefix": "__ERROR__:"},
    )
    assert failed_by_marker[0] == "failed"

    failed_by_tool = evaluate_failure_by_policy(
        "normal text",
        diagnostics={"threshold_hit": True, "threshold_reason": "too many tool errors"},
        policy={"mode": "tool_only", "error_marker_prefix": "__ERROR__:"},
    )
    assert failed_by_tool[0] == "failed"

    failed_by_hybrid = evaluate_failure_by_policy(
        "__ERROR__: invalid result",
        diagnostics={"threshold_hit": False},
        policy={"mode": "hybrid", "error_marker_prefix": "__ERROR__:"},
    )
    assert failed_by_hybrid[0] == "failed"
