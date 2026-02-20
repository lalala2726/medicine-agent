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
        messages: list[HumanMessage] | None = None,
) -> dict:
    history_messages = messages or [HumanMessage(content="帮我处理复杂任务")]
    return {
        "user_input": "帮我处理复杂任务",
        "messages": history_messages,
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
            return _DummyResponse(
                json.dumps(
                    {
                        "next_node": "product_agent",
                        "directive": "查询商品ID=2001库存并返回状态",
                    }
                )
            )

    monkeypatch.setattr(supervisor_module, "create_chat_model", lambda **_kwargs: _Model())
    result = supervisor_module.supervisor_agent(_build_state())
    assert result["next_node"] == "product_agent"
    assert result["routing"]["finished"] is False
    assert result["routing"]["directive"] == "查询商品ID=2001库存并返回状态"
    assert result["context"]["node_call_counts"]["product_agent"] == 1


def test_supervisor_falls_back_to_finish_on_invalid_next_node(monkeypatch: pytest.MonkeyPatch):
    class _Model:
        def invoke(self, _messages):
            return _DummyResponse(json.dumps({"next_node": "bad_node", "directive": "忽略"}))

    monkeypatch.setattr(supervisor_module, "create_chat_model", lambda **_kwargs: _Model())
    result = supervisor_module.supervisor_agent(_build_state())
    assert result["next_node"] == "FINISH"
    assert result["routing"]["finished"] is True
    assert result["routing"]["directive"] == ""


def test_supervisor_falls_back_to_finish_when_directive_missing(monkeypatch: pytest.MonkeyPatch):
    class _Model:
        def invoke(self, _messages):
            return _DummyResponse(json.dumps({"next_node": "order_agent"}))

    monkeypatch.setattr(supervisor_module, "create_chat_model", lambda **_kwargs: _Model())
    result = supervisor_module.supervisor_agent(_build_state())
    assert result["next_node"] == "FINISH"
    assert result["routing"]["finished"] is True
    assert result["routing"]["directive"] == ""


def test_supervisor_loop_guard_finishes_when_same_node_reaches_limit(
        monkeypatch: pytest.MonkeyPatch,
):
    class _Model:
        def invoke(self, _messages):
            return _DummyResponse(
                json.dumps(
                    {
                        "next_node": "order_agent",
                        "directive": "查询订单123的退款记录",
                    }
                )
            )

    monkeypatch.setattr(supervisor_module, "create_chat_model", lambda **_kwargs: _Model())
    result = supervisor_module.supervisor_agent(
        _build_state(counts={"order_agent": 2}, turn=1)
    )
    assert result["next_node"] == "FINISH"
    assert result["routing"]["finished"] is True
    assert result["routing"]["directive"] == ""
    assert result["context"]["node_call_counts"]["order_agent"] == 2


def test_supervisor_uses_full_messages_instead_of_tail(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class _Model:
        def invoke(self, messages):
            captured["messages"] = messages
            return _DummyResponse(
                json.dumps(
                    {
                        "next_node": "product_agent",
                        "directive": "检查商品2001",
                    }
                )
            )

    monkeypatch.setattr(supervisor_module, "create_chat_model", lambda **_kwargs: _Model())
    full_history = [HumanMessage(content=f"历史消息{i}") for i in range(12)]
    supervisor_module.supervisor_agent(_build_state(messages=full_history))

    input_messages = captured["messages"]
    assert isinstance(input_messages, list)
    assert len(input_messages) == 2
    payload = json.loads(str(input_messages[1].content))
    assert len(payload["messages"]) == 12
