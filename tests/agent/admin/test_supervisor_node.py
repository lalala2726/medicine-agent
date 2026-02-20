import json

import pytest
from langchain_core.messages import HumanMessage

import app.agent.admin.node.supervisor_node as supervisor_module
from app.agent.admin.model_policy import DEFAULT_NODE_GOAL


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
            "task_difficulty": "normal",
        },
        "results": {},
        "execution_traces": [],
        "errors": [],
    }


def test_supervisor_returns_valid_target_node(monkeypatch: pytest.MonkeyPatch):
    class _Model:
        def invoke(self, _messages):
            return _DummyResponse(
                json.dumps(
                    {
                        "target_node": "product_agent",
                        "node_goal": "查询商品ID=2001库存并返回状态",
                        "task_difficulty": "complex",
                    }
                )
            )

    monkeypatch.setattr(supervisor_module, "create_chat_model", lambda **_kwargs: _Model())
    result = supervisor_module.supervisor_agent(_build_state())
    assert result["routing"]["target_node"] == "product_agent"
    assert result["routing"]["finished"] is False
    assert result["routing"]["node_goal"] == "查询商品ID=2001库存并返回状态"
    assert result["routing"]["task_difficulty"] == "complex"
    assert result["routing"]["selected_model"] == "qwen-max"
    assert result["routing"]["think_enabled"] is True
    assert result["context"]["node_call_counts"]["product_agent"] == 1


def test_supervisor_falls_back_to_summary_on_invalid_target(monkeypatch: pytest.MonkeyPatch):
    class _Model:
        def invoke(self, _messages):
            return _DummyResponse(
                json.dumps(
                    {
                        "target_node": "bad_node",
                        "node_goal": "忽略",
                        "task_difficulty": "simple",
                    }
                )
            )

    monkeypatch.setattr(supervisor_module, "create_chat_model", lambda **_kwargs: _Model())
    result = supervisor_module.supervisor_agent(_build_state())
    assert result["routing"]["target_node"] == "summary_agent"
    assert result["routing"]["finished"] is True
    assert result["routing"]["node_goal"] == "忽略"
    assert result["routing"]["route_target"] == "summary_agent"
    assert result["routing"]["task_difficulty"] == "simple"
    assert result["routing"]["selected_model"] == "qwen-flash"
    assert result["routing"]["think_enabled"] is False


def test_supervisor_fills_default_goal_when_node_goal_missing(monkeypatch: pytest.MonkeyPatch):
    class _Model:
        def invoke(self, _messages):
            return _DummyResponse(
                json.dumps(
                    {
                        "target_node": "order_agent",
                        "task_difficulty": "complex",
                    }
                )
            )

    monkeypatch.setattr(supervisor_module, "create_chat_model", lambda **_kwargs: _Model())
    result = supervisor_module.supervisor_agent(_build_state())
    assert result["routing"]["target_node"] == "order_agent"
    assert result["routing"]["finished"] is False
    assert result["routing"]["node_goal"] == DEFAULT_NODE_GOAL
    assert result["routing"]["task_difficulty"] == "complex"
    assert result["routing"]["selected_model"] == "qwen-max"
    assert result["routing"]["think_enabled"] is True
    assert result["context"]["node_call_counts"]["order_agent"] == 1


def test_supervisor_loop_guard_routes_to_summary_when_limit_reached(
        monkeypatch: pytest.MonkeyPatch,
):
    class _Model:
        def invoke(self, _messages):
            return _DummyResponse(
                json.dumps(
                    {
                        "target_node": "order_agent",
                        "node_goal": "查询订单123的退款记录",
                        "task_difficulty": "complex",
                    }
                )
            )

    monkeypatch.setattr(supervisor_module, "create_chat_model", lambda **_kwargs: _Model())
    result = supervisor_module.supervisor_agent(
        _build_state(counts={"order_agent": 2}, turn=1)
    )
    assert result["routing"]["target_node"] == "summary_agent"
    assert result["routing"]["finished"] is True
    assert result["routing"]["node_goal"] == DEFAULT_NODE_GOAL
    assert result["routing"]["route_target"] == "summary_agent"
    assert result["routing"]["task_difficulty"] == "complex"
    assert result["routing"]["selected_model"] == "qwen-max"
    assert result["routing"]["think_enabled"] is True
    assert result["context"]["node_call_counts"]["order_agent"] == 2


def test_supervisor_uses_full_messages_instead_of_tail(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class _Model:
        def invoke(self, messages):
            captured["messages"] = messages
            return _DummyResponse(
                json.dumps(
                    {
                        "target_node": "product_agent",
                        "node_goal": "检查商品2001",
                        "task_difficulty": "normal",
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


def test_supervisor_falls_back_to_existing_task_difficulty_when_missing(monkeypatch: pytest.MonkeyPatch):
    class _Model:
        def invoke(self, _messages):
            return _DummyResponse(
                json.dumps(
                    {
                        "target_node": "product_agent",
                        "node_goal": "查询商品2001库存",
                    }
                )
            )

    monkeypatch.setattr(supervisor_module, "create_chat_model", lambda **_kwargs: _Model())
    state = _build_state()
    state["routing"]["task_difficulty"] = "complex"
    result = supervisor_module.supervisor_agent(state)
    assert result["routing"]["task_difficulty"] == "complex"
    assert result["routing"]["selected_model"] == "qwen-max"
    assert result["routing"]["think_enabled"] is True


def test_supervisor_emits_enter_and_dispatch_notice_when_emitter_available(
        monkeypatch: pytest.MonkeyPatch,
):
    emitted = []

    class _Model:
        def invoke(self, _messages):
            return _DummyResponse(
                json.dumps(
                    {
                        "target_node": "product_agent",
                        "node_goal": "查询商品2001库存并返回状态",
                        "task_difficulty": "normal",
                    }
                )
            )

    monkeypatch.setattr(supervisor_module, "create_chat_model", lambda **_kwargs: _Model())
    monkeypatch.setattr(supervisor_module, "has_status_emitter", lambda: True)
    monkeypatch.setattr(supervisor_module, "emit_sse_response", lambda payload: emitted.append(payload))

    supervisor_module.supervisor_agent(_build_state())

    assert len(emitted) == 2
    assert emitted[0].content.node == "supervisor_agent"
    assert emitted[0].content.state == "start"
    assert emitted[0].type.value == "notice"

    dispatch = emitted[1]
    assert dispatch.content.node == "supervisor_agent"
    assert dispatch.content.state == "dispatch"
    assert dispatch.content.name == "product_agent"
    assert dispatch.content.arguments is None
    assert dispatch.meta["target_node"] == "product_agent"
    assert "node_goal" not in dispatch.meta
