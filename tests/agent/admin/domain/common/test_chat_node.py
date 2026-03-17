from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.agent.admin.domain.chat import chat_node as node_module


def _patch_chat_agent_dependencies(
        monkeypatch: pytest.MonkeyPatch,
        *,
        knowledge_enabled: bool,
) -> tuple[dict[str, object], object, object, object, object, object]:
    captured: dict[str, object] = {}
    knowledge_tool = object()
    safe_user_tool = object()
    base_prompt_middleware = object()
    tool_status_middleware = object()
    skill_middleware = object()

    monkeypatch.setattr(
        node_module,
        "get_current_agent_config_snapshot",
        lambda: SimpleNamespace(is_knowledge_enabled=lambda: knowledge_enabled),
    )
    monkeypatch.setattr(node_module, "search_knowledge_context", knowledge_tool)
    monkeypatch.setattr(node_module, "get_safe_user_info", safe_user_tool)
    monkeypatch.setattr(
        node_module,
        "append_current_time_to_prompt",
        lambda prompt: f"{prompt}\n\n当前时间：2026-03-15 09:30 UTC+8",
    )
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
    return (
        captured,
        knowledge_tool,
        safe_user_tool,
        base_prompt_middleware,
        tool_status_middleware,
        skill_middleware,
    )


def test_chat_agent_registers_knowledge_tool_and_tool_status_middleware(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        captured,
        knowledge_tool,
        safe_user_tool,
        base_prompt_middleware,
        tool_status_middleware,
        skill_middleware,
    ) = _patch_chat_agent_dependencies(
        monkeypatch,
        knowledge_enabled=True,
    )

    result = node_module.chat_agent(
        {
            "history_messages": [],
            "execution_traces": [],
        }
    )

    assert captured["tools"] == [
        knowledge_tool,
        safe_user_tool,
    ]
    assert "不是单纯的闲聊节点" in captured["system_prompt"].content
    assert "必须先调用知识库检索工具" in captured["system_prompt"].content
    assert "当前时间：2026-03-15 09:30 UTC+8" in captured["system_prompt"].content
    assert captured["middleware"] == [
        base_prompt_middleware,
        tool_status_middleware,
        skill_middleware,
    ]
    assert result["result"] == "知识库回答"
    assert result["messages"][0].content == "知识库回答"


def test_chat_agent_skips_knowledge_tool_when_knowledge_disabled(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        captured,
        knowledge_tool,
        safe_user_tool,
        base_prompt_middleware,
        tool_status_middleware,
        skill_middleware,
    ) = _patch_chat_agent_dependencies(
        monkeypatch,
        knowledge_enabled=False,
    )

    result = node_module.chat_agent(
        {
            "history_messages": [],
            "execution_traces": [],
        }
    )

    assert knowledge_tool not in captured["tools"]
    assert captured["tools"] == [
        safe_user_tool,
    ]
    assert "当前环境未启用知识库问答能力" in captured["system_prompt"].content
    assert "必须先调用知识库检索工具" not in captured["system_prompt"].content
    assert "当前时间：2026-03-15 09:30 UTC+8" in captured["system_prompt"].content
    assert captured["middleware"] == [
        base_prompt_middleware,
        tool_status_middleware,
        skill_middleware,
    ]
    assert result["result"] == "知识库回答"
