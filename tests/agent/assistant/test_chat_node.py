from __future__ import annotations

import importlib
from types import SimpleNamespace

from langchain_core.messages import AIMessage

chat_node = importlib.import_module("app.agent.assistant.node.chat_node")


def test_chat_agent_prefers_llm_model_name_when_present(
        monkeypatch,
) -> None:
    """测试目的：验证 chat 节点优先使用 llm.model_name；预期结果：execution_trace.model_name 等于 llm.model_name。"""

    captured: dict[str, object] = {}
    fake_agent = object()
    fake_llm = SimpleNamespace(model_name="resolved-model-from-llm")

    monkeypatch.setattr(chat_node, "create_chat_model", lambda **_kwargs: fake_llm)
    monkeypatch.setattr(chat_node, "create_agent", lambda **_kwargs: fake_agent)
    monkeypatch.setattr(chat_node, "agent_stream", lambda *_args, **_kwargs: {"streamed_text": "AI回答"})
    monkeypatch.setattr(
        chat_node,
        "record_agent_trace",
        lambda **_kwargs: {
            "text": "AI回答",
            "model_name": "trace-model-name",
            "is_usage_complete": True,
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        },
    )

    def _fake_append_trace_and_refresh_token_usage(_execution_traces, trace_item):
        captured["trace_item"] = trace_item
        return [trace_item], {"total_tokens": 3}

    monkeypatch.setattr(
        chat_node,
        "append_trace_and_refresh_token_usage",
        _fake_append_trace_and_refresh_token_usage,
    )

    result = chat_node.chat_agent({"history_messages": []})

    assert result["result"] == "AI回答"
    assert isinstance(result["messages"][0], AIMessage)
    assert captured["trace_item"]["model_name"] == "resolved-model-from-llm"


def test_chat_agent_falls_back_to_trace_model_name_when_llm_model_name_missing(
        monkeypatch,
) -> None:
    """测试目的：验证 llm.model_name 缺失时回退 trace.model_name；预期结果：execution_trace.model_name 等于 trace.model_name。"""

    captured: dict[str, object] = {}
    fake_agent = object()
    fake_llm = SimpleNamespace(model_name="")

    monkeypatch.setattr(chat_node, "create_chat_model", lambda **_kwargs: fake_llm)
    monkeypatch.setattr(chat_node, "create_agent", lambda **_kwargs: fake_agent)
    monkeypatch.setattr(chat_node, "agent_stream", lambda *_args, **_kwargs: {"streamed_text": "AI回答"})
    monkeypatch.setattr(
        chat_node,
        "record_agent_trace",
        lambda **_kwargs: {
            "text": "AI回答",
            "model_name": "trace-model-name",
            "is_usage_complete": True,
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        },
    )

    def _fake_append_trace_and_refresh_token_usage(_execution_traces, trace_item):
        captured["trace_item"] = trace_item
        return [trace_item], {"total_tokens": 3}

    monkeypatch.setattr(
        chat_node,
        "append_trace_and_refresh_token_usage",
        _fake_append_trace_and_refresh_token_usage,
    )

    chat_node.chat_agent({"history_messages": []})

    assert captured["trace_item"]["model_name"] == "trace-model-name"


def test_chat_agent_uses_unknown_when_llm_and_trace_model_name_missing(
        monkeypatch,
) -> None:
    """测试目的：验证 llm/trace 都无模型名时回退 unknown；预期结果：execution_trace.model_name 为 unknown。"""

    captured: dict[str, object] = {}
    fake_agent = object()
    fake_llm = SimpleNamespace(model_name="")

    monkeypatch.setattr(chat_node, "create_chat_model", lambda **_kwargs: fake_llm)
    monkeypatch.setattr(chat_node, "create_agent", lambda **_kwargs: fake_agent)
    monkeypatch.setattr(chat_node, "agent_stream", lambda *_args, **_kwargs: {"streamed_text": "AI回答"})
    monkeypatch.setattr(
        chat_node,
        "record_agent_trace",
        lambda **_kwargs: {
            "text": "AI回答",
            "model_name": "",
            "is_usage_complete": True,
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        },
    )

    def _fake_append_trace_and_refresh_token_usage(_execution_traces, trace_item):
        captured["trace_item"] = trace_item
        return [trace_item], {"total_tokens": 3}

    monkeypatch.setattr(
        chat_node,
        "append_trace_and_refresh_token_usage",
        _fake_append_trace_and_refresh_token_usage,
    )

    chat_node.chat_agent({"history_messages": []})

    assert captured["trace_item"]["model_name"] == "unknown"
