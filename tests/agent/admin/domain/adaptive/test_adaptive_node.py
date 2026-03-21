from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.agent.admin.domain.adaptive import adaptive_node as node_module


def test_build_adaptive_tools_keeps_user_info_as_only_base_tool(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    domain_tool = object()
    safe_user_tool = object()

    monkeypatch.setattr(
        node_module,
        "_DOMAIN_TOOL_MAP",
        {"order_agent": (domain_tool,)},
    )
    monkeypatch.setattr(node_module, "_BASE_ADAPTIVE_TOOLS", (safe_user_tool,))

    assert node_module._build_adaptive_tools(["order_agent"]) == [
        domain_tool,
        safe_user_tool,
    ]
    assert node_module._build_adaptive_tools([]) == [safe_user_tool]


def test_adaptive_agent_injects_current_time_into_system_prompt(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    domain_tool = object()
    safe_user_tool = object()
    base_prompt_middleware = object()
    tool_status_middleware = object()
    skill_middleware = object()
    tool_limit_middleware = object()

    monkeypatch.setattr(node_module, "model_switch", lambda _state: "adaptive-slot")
    monkeypatch.setattr(
        node_module,
        "_DOMAIN_TOOL_MAP",
        {"order_agent": (domain_tool,)},
    )
    monkeypatch.setattr(node_module, "_BASE_ADAPTIVE_TOOLS", (safe_user_tool,))
    monkeypatch.setattr(
        node_module,
        "append_current_time_to_prompt",
        lambda prompt: f"{prompt}\n\n当前时间：2026-03-15 09:30 UTC+8",
    )
    monkeypatch.setattr(
        node_module,
        "create_agent_chat_llm",
        lambda **_kwargs: SimpleNamespace(model_name="adaptive-model"),
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
        "ToolCallLimitMiddleware",
        lambda **_kwargs: tool_limit_middleware,
    )
    monkeypatch.setattr(
        node_module,
        "create_agent",
        lambda **kwargs: captured.update(kwargs) or object(),
    )
    monkeypatch.setattr(
        node_module,
        "agent_stream",
        lambda *_args, **_kwargs: {"streamed_text": "自适应回答"},
    )
    monkeypatch.setattr(
        node_module,
        "record_agent_trace",
        lambda **_kwargs: {
            "text": "自适应回答",
            "model_name": "adaptive-model",
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

    result = node_module.adaptive_agent(
        {
            "routing": {
                "route_targets": ["order_agent"],
            },
            "history_messages": [],
            "execution_traces": [],
        }
    )

    assert captured["tools"] == [domain_tool, safe_user_tool]
    assert "当前时间：2026-03-15 09:30 UTC+8" in captured["system_prompt"].content
    assert captured["middleware"] == [
        base_prompt_middleware,
        tool_status_middleware,
        skill_middleware,
        tool_limit_middleware,
    ]
    assert result["result"] == "自适应回答"
    assert result["messages"][0].content == "自适应回答"
