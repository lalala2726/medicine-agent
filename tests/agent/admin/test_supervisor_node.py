import json

import pytest
from langchain_core.messages import HumanMessage

import app.agent.admin.node.supervisor_node as supervisor_module


class _DummyResponse:
    def __init__(self, content: str):
        self.content = content


def _build_state(
        *,
        counts: dict[str, int] | None = None,
        turn: int = 0,
) -> dict:
    return {
        "user_input": "帮我处理复杂任务",
        "messages": [HumanMessage(content="帮我处理复杂任务")],
        "context": {
            "node_call_counts": counts or {},
            "agent_outputs": {},
        },
        "routing": {
            "mode": "supervisor_loop",
            "turn": turn,
        },
        "results": {},
        "execution_traces": [],
        "errors": [],
    }


def test_supervisor_returns_valid_next_node(monkeypatch: pytest.MonkeyPatch):
    class _Model:
        def invoke(self, _messages):
            return _DummyResponse(json.dumps({"next_node": "product_agent"}))

    monkeypatch.setattr(supervisor_module, "create_chat_model", lambda **_kwargs: _Model())
    result = supervisor_module.supervisor_agent(_build_state())
    assert result["next_node"] == "product_agent"
    assert result["routing"]["finished"] is False
    assert result["context"]["node_call_counts"]["product_agent"] == 1


def test_supervisor_falls_back_to_finish_on_invalid_next_node(monkeypatch: pytest.MonkeyPatch):
    class _Model:
        def invoke(self, _messages):
            return _DummyResponse(json.dumps({"next_node": "bad_node"}))

    monkeypatch.setattr(supervisor_module, "create_chat_model", lambda **_kwargs: _Model())
    result = supervisor_module.supervisor_agent(_build_state())
    assert result["next_node"] == "FINISH"
    assert result["routing"]["finished"] is True


def test_supervisor_loop_guard_finishes_when_same_node_reaches_limit(
        monkeypatch: pytest.MonkeyPatch,
):
    class _Model:
        def invoke(self, _messages):
            return _DummyResponse(json.dumps({"next_node": "order_agent"}))

    monkeypatch.setattr(supervisor_module, "create_chat_model", lambda **_kwargs: _Model())
    result = supervisor_module.supervisor_agent(
        _build_state(counts={"order_agent": 2}, turn=1)
    )
    assert result["next_node"] == "FINISH"
    assert result["routing"]["finished"] is True
    assert result["context"]["node_call_counts"]["order_agent"] == 2

