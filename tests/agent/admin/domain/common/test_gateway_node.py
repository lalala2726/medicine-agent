from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from app.agent.admin.domain.common import gateway_node as node_module


def test_gateway_router_prompt_routes_knowledge_questions_to_chat_agent(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    base_prompt_middleware = object()

    monkeypatch.setattr(
        node_module,
        "create_agent_chat_llm",
        lambda **_kwargs: SimpleNamespace(model_name="gateway-model"),
    )
    monkeypatch.setattr(node_module, "BasePromptMiddleware", lambda: base_prompt_middleware)
    monkeypatch.setattr(
        node_module,
        "create_agent",
        lambda **kwargs: captured.update(kwargs) or object(),
    )
    monkeypatch.setattr(
        node_module,
        "agent_invoke",
        lambda *_args, **_kwargs: SimpleNamespace(
            payload={
                "messages": [
                    AIMessage(content='{"route_targets":["chat_agent"],"task_difficulty":"normal"}')
                ]
            },
            content='{"route_targets":["chat_agent"],"task_difficulty":"normal"}',
        ),
    )
    monkeypatch.setattr(
        node_module,
        "record_agent_trace",
        lambda **_kwargs: {
            "text": '{"route_targets":["chat_agent"],"task_difficulty":"normal"}',
            "model_name": "gateway-model",
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

    result = node_module.gateway_router(
        {
            "history_messages": [],
            "execution_traces": [],
        }
    )

    assert "知识库型问题" in captured["system_prompt"].content
    assert "必须输出 `[\"chat_agent\"]`" in captured["system_prompt"].content
    assert "不要因为问题里出现“订单”“商品”“用户”等关键词，就误判为业务节点" in captured["system_prompt"].content
    assert "response_format" not in captured
    assert captured["middleware"] == [base_prompt_middleware]
    assert result["routing"] == {"route_targets": ["chat_agent"], "task_difficulty": "normal"}


def test_resolve_gateway_routing_result_parses_json_from_messages() -> None:
    payload = {
        "messages": [
            AIMessage(content='{"route_targets":["order_agent"],"task_difficulty":"normal"}')
        ]
    }

    routing = node_module._resolve_gateway_routing_result(payload)

    assert routing == {"route_targets": ["order_agent"], "task_difficulty": "normal"}


def test_resolve_gateway_routing_result_falls_back_when_message_is_not_json() -> None:
    payload = {
        "messages": [
            AIMessage(content="我觉得应该走 order_agent")
        ]
    }

    routing = node_module._resolve_gateway_routing_result(payload)

    assert routing == {"route_targets": ["chat_agent"], "task_difficulty": "normal"}
