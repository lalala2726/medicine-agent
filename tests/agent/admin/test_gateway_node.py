import json

import pytest

import app.agent.admin.node.gateway_node as gateway_module


class _DummyResponse:
    def __init__(self, content: str):
        self.content = content


def _build_state(user_input: str) -> dict:
    return {
        "user_input": user_input,
        "messages": [],
        "context": {},
        "routing": {},
        "results": {},
        "execution_traces": [],
        "errors": [],
    }


def test_gateway_routes_order_agent(monkeypatch: pytest.MonkeyPatch):
    class _Model:
        def invoke(self, _messages):
            return _DummyResponse(
                json.dumps(
                    {
                        "route_target": "order_agent",
                        "task_difficulty": "simple",
                    }
                )
            )

    monkeypatch.setattr(gateway_module, "create_chat_model", lambda **_kwargs: _Model())
    result = gateway_module.gateway_router(_build_state("查一下订单123"))
    assert result["routing"]["route_target"] == "order_agent"
    assert result["routing"]["mode"] == "fast_lane"
    assert result["routing"]["task_difficulty"] == "simple"
    assert result["routing"]["selected_model"] == "qwen-flash"
    assert result["routing"]["think_enabled"] is False
    assert result["next_node"] == "order_agent"


def test_gateway_routes_chat_agent(monkeypatch: pytest.MonkeyPatch):
    class _Model:
        def invoke(self, _messages):
            return _DummyResponse(
                json.dumps(
                    {
                        "route_target": "chat_agent",
                        "task_difficulty": "normal",
                    }
                )
            )

    monkeypatch.setattr(gateway_module, "create_chat_model", lambda **_kwargs: _Model())
    result = gateway_module.gateway_router(_build_state("在吗"))
    assert result["routing"]["route_target"] == "chat_agent"
    assert result["routing"]["mode"] == "chat"
    assert result["routing"]["task_difficulty"] == "normal"
    assert result["routing"]["selected_model"] == "qwen-plus"
    assert result["routing"]["think_enabled"] is False
    assert result["next_node"] == "chat_agent"


def test_gateway_falls_back_to_supervisor_on_invalid_target(monkeypatch: pytest.MonkeyPatch):
    class _Model:
        def invoke(self, _messages):
            return _DummyResponse(json.dumps({"route_target": "unknown"}))

    monkeypatch.setattr(gateway_module, "create_chat_model", lambda **_kwargs: _Model())
    result = gateway_module.gateway_router(_build_state("把退款订单对应商品下架"))
    assert result["routing"]["route_target"] == "supervisor_agent"
    assert result["routing"]["mode"] == "supervisor_loop"
    assert result["routing"]["task_difficulty"] == "normal"
    assert result["routing"]["selected_model"] == "qwen-plus"
    assert result["routing"]["think_enabled"] is False
    assert result["next_node"] == "supervisor_agent"


def test_gateway_falls_back_to_supervisor_on_model_error(monkeypatch: pytest.MonkeyPatch):
    class _Model:
        def invoke(self, _messages):
            raise RuntimeError("boom")

    monkeypatch.setattr(gateway_module, "create_chat_model", lambda **_kwargs: _Model())
    result = gateway_module.gateway_router(_build_state("复杂任务"))
    assert result["routing"]["route_target"] == "supervisor_agent"
    assert result["routing"]["task_difficulty"] == "normal"
    assert result["routing"]["selected_model"] == "qwen-plus"
    assert result["routing"]["think_enabled"] is False
    assert result["next_node"] == "supervisor_agent"


def test_gateway_normalizes_invalid_task_difficulty(monkeypatch: pytest.MonkeyPatch):
    class _Model:
        def invoke(self, _messages):
            return _DummyResponse(
                json.dumps(
                    {
                        "route_target": "supervisor_agent",
                        "task_difficulty": "super-complex",
                    }
                )
            )

    monkeypatch.setattr(gateway_module, "create_chat_model", lambda **_kwargs: _Model())
    result = gateway_module.gateway_router(_build_state("复杂任务"))
    assert result["routing"]["task_difficulty"] == "normal"
    assert result["routing"]["selected_model"] == "qwen-plus"
