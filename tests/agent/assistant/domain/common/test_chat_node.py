from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.agent.assistant.domain.common import chat_node as node_module


def test_chat_agent_registers_knowledge_tool_and_tool_status_middleware(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    knowledge_tool = object()
    current_time_tool = object()
    safe_user_tool = object()
    base_prompt_middleware = object()
    tool_status_middleware = object()
    skill_middleware = object()

    monkeypatch.setattr(node_module, "search_knowledge_context", knowledge_tool)
    monkeypatch.setattr(node_module, "get_current_time", current_time_tool)
    monkeypatch.setattr(node_module, "get_safe_user_info", safe_user_tool)
    monkeypatch.setattr(
        node_module,
        "create_agent_chat_llm",
        lambda **_kwargs: SimpleNamespace(model_name="chat-model"),
    )
    monkeypatch.setattr(node_module, "BasePromptMiddleware", lambda: base_prompt_middleware)
    monkeypatch.setattr(
        node_module,
        "build_tool_status_middleware",
        lambda: tool_status_middleware,
    )
    monkeypatch.setattr(
        node_module,
        "SkillMiddleware",
        lambda **_kwargs: skill_middleware,
    )
    monkeypatch.setattr(
        node_module,
        "create_agent",
        lambda **kwargs: captured.update(kwargs) or object(),
    )
    monkeypatch.setattr(
        node_module,
        "agent_stream",
        lambda *_args, **_kwargs: {"streamed_text": "知识库回答"},
    )
    monkeypatch.setattr(
        node_module,
        "record_agent_trace",
        lambda **_kwargs: {
            "text": "知识库回答",
            "model_name": "chat-model",
            "is_usage_complete": True,
            "usage": None,
        },
    )
    monkeypatch.setattr(
        node_module,
        "append_trace_and_refresh_token_usage",
        lambda current_execution_traces, trace_item: (
            current_execution_traces + [trace_item],
            None,
        ),
    )

    result = node_module.chat_agent(
        {
            "history_messages": [],
            "execution_traces": [],
        }
    )

    assert captured["tools"] == [
        knowledge_tool,
        current_time_tool,
        safe_user_tool,
    ]
    assert "不是单纯的闲聊节点" in captured["system_prompt"].content
    assert "必须先调用知识库检索工具" in captured["system_prompt"].content
    assert captured["middleware"] == [
        base_prompt_middleware,
        tool_status_middleware,
        skill_middleware,
    ]
    assert result["result"] == "知识库回答"
    assert result["messages"][0].content == "知识库回答"
