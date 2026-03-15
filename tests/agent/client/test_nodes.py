from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from app.agent.client import after_sale_node as after_sale_module
from app.agent.client import chat_node as chat_module
from app.agent.client import gateway_node as gateway_module
from app.agent.client import workflow as workflow_module


def test_route_from_gateway_returns_after_sale_for_valid_target():
    result = workflow_module._route_from_gateway(
        {"routing": {"route_targets": ["after_sale_agent"]}}
    )

    assert result == "after_sale_agent"


def test_route_from_gateway_falls_back_to_chat_for_multiple_targets():
    result = workflow_module._route_from_gateway(
        {"routing": {"route_targets": ["chat_agent", "after_sale_agent"]}}
    )

    assert result == "chat_agent"


def test_gateway_router_resolves_after_sale_route(monkeypatch):
    monkeypatch.setattr(
        gateway_module,
        "create_agent_chat_llm",
        lambda **_kwargs: SimpleNamespace(model_name="mock-route-model"),
    )
    monkeypatch.setattr(gateway_module, "create_agent", lambda **_kwargs: object())
    monkeypatch.setattr(
        gateway_module,
        "agent_invoke",
        lambda _agent, _messages: SimpleNamespace(
            payload={
                "messages": [
                    SimpleNamespace(
                        content='{"route_targets":["after_sale_agent"],"task_difficulty":"high"}'
                    )
                ]
            },
            content="",
        ),
    )
    monkeypatch.setattr(
        gateway_module,
        "record_agent_trace",
        lambda **_kwargs: {
            "text": '{"route_targets":["after_sale_agent"],"task_difficulty":"high"}',
            "is_usage_complete": True,
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        },
    )

    result = gateway_module.gateway_router(
        {
            "history_messages": [HumanMessage(content="我要申请退款")],
            "execution_traces": [],
        }
    )

    assert result["routing"] == {
        "route_targets": ["after_sale_agent"],
        "task_difficulty": "high",
    }
    assert result["execution_traces"][0]["node_name"] == "gateway_router"
    assert result["token_usage"]["total_tokens"] == 2


def test_gateway_router_falls_back_to_chat_on_invalid_target(monkeypatch):
    monkeypatch.setattr(
        gateway_module,
        "create_agent_chat_llm",
        lambda **_kwargs: SimpleNamespace(model_name="mock-route-model"),
    )
    monkeypatch.setattr(gateway_module, "create_agent", lambda **_kwargs: object())
    monkeypatch.setattr(
        gateway_module,
        "agent_invoke",
        lambda _agent, _messages: SimpleNamespace(
            payload={
                "messages": [
                    SimpleNamespace(
                        content='{"route_targets":["unknown_agent"],"task_difficulty":"normal"}'
                    )
                ]
            },
            content="",
        ),
    )
    monkeypatch.setattr(
        gateway_module,
        "record_agent_trace",
        lambda **_kwargs: {
            "text": "",
            "is_usage_complete": True,
            "usage": None,
        },
    )

    result = gateway_module.gateway_router(
        {
            "history_messages": [HumanMessage(content="随便问问")],
            "execution_traces": [],
        }
    )

    assert result["routing"] == {
        "route_targets": ["chat_agent"],
        "task_difficulty": "normal",
    }


def test_chat_agent_appends_trace_and_token_usage(monkeypatch):
    monkeypatch.setattr(
        chat_module,
        "create_agent_chat_llm",
        lambda **_kwargs: SimpleNamespace(model_name="mock-chat-model"),
    )
    monkeypatch.setattr(chat_module, "create_agent", lambda **_kwargs: object())
    monkeypatch.setattr(
        chat_module,
        "agent_stream",
        lambda *_args, **_kwargs: {"streamed_text": "您好，我来帮您"},
    )
    monkeypatch.setattr(
        chat_module,
        "record_agent_trace",
        lambda **_kwargs: {
            "text": "您好，我来帮您",
            "model_name": "mock-chat-model",
            "is_usage_complete": True,
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "total_tokens": 5,
            },
        },
    )

    result = chat_module.chat_agent(
        {
            "history_messages": [HumanMessage(content="你好")],
            "execution_traces": [],
        }
    )

    assert result["result"] == "您好，我来帮您"
    assert result["execution_traces"][0]["node_name"] == "chat_agent"
    assert result["token_usage"]["total_tokens"] == 5


def test_after_sale_agent_appends_trace_and_token_usage(monkeypatch):
    monkeypatch.setattr(
        after_sale_module,
        "create_agent_chat_llm",
        lambda **_kwargs: SimpleNamespace(model_name="mock-after-sale-model"),
    )
    monkeypatch.setattr(after_sale_module, "create_agent", lambda **_kwargs: object())
    monkeypatch.setattr(
        after_sale_module,
        "agent_stream",
        lambda *_args, **_kwargs: {"streamed_text": "请先提供订单号和问题照片"},
    )
    monkeypatch.setattr(
        after_sale_module,
        "record_agent_trace",
        lambda **_kwargs: {
            "text": "请先提供订单号和问题照片",
            "model_name": "mock-after-sale-model",
            "is_usage_complete": True,
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 4,
                "total_tokens": 6,
            },
        },
    )

    result = after_sale_module.after_sale_agent(
        {
            "routing": {"task_difficulty": "normal"},
            "history_messages": [HumanMessage(content="我想退款")],
            "execution_traces": [],
        }
    )

    assert result["result"] == "请先提供订单号和问题照片"
    assert result["execution_traces"][0]["node_name"] == "after_sale_agent"
    assert result["token_usage"]["total_tokens"] == 6
