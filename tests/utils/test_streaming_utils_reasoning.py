from __future__ import annotations

import app.utils.streaming_utils as streaming_utils


class _Chunk:
    def __init__(self, content: str = "", reasoning: str = "") -> None:
        self.content = content
        self.additional_kwargs = {}
        if reasoning:
            self.additional_kwargs["reasoning_content"] = reasoning


class _DummyLLM:
    def stream(self, _messages):
        yield _Chunk(reasoning="先分析问题")
        yield _Chunk(content="最终答案")


def test_stream_with_reasoning_separates_answer_and_reasoning():
    answer_chunks, reasoning_chunks = streaming_utils.stream_with_reasoning(
        _DummyLLM(),
        messages=[{"role": "user", "content": "你好"}],
    )

    assert answer_chunks == ["最终答案"]
    assert reasoning_chunks == ["先分析问题"]


def test_invoke_with_policy_exposes_reasoning_chunks_in_diagnostics():
    content, diagnostics = streaming_utils.invoke_with_policy(
        _DummyLLM(),
        [{"role": "user", "content": "你好"}],
        enable_stream=True,
    )

    assert content == "最终答案"
    assert diagnostics["stream_chunks"] == ["最终答案"]
    assert diagnostics["reasoning_chunks"] == ["先分析问题"]
