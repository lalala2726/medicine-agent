import app.utils.streaming_utils as streaming_utils_module


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
