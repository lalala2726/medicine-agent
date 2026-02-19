from langchain_core.messages import AIMessage, HumanMessage

import app.agent.admin.node.chat_node as chat_module


class _DummyResponse:
    def __init__(self, content: str):
        self.content = content


class _DummyModel:
    def __init__(self):
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return _DummyResponse("ok")


def _build_state(
        *,
        fallback_context: dict | None = None,
        history_messages: list[HumanMessage | AIMessage] | None = None,
) -> dict:
    routing: dict = {"route_target": "coordinator_agent"}
    if fallback_context is not None:
        routing["fallback_context"] = fallback_context
    return {
        "user_input": "你好",
        "user_intent": {},
        "plan": [],
        "routing": routing,
        "history_messages": history_messages or [],
        "step_outputs": {},
        "shared_memory": {},
        "results": {},
        "errors": [],
    }


def test_chat_agent_uses_fallback_mode_when_fallback_context_exists(monkeypatch):
    model = _DummyModel()
    monkeypatch.setattr(chat_module, "create_chat_model", lambda *args, **kwargs: model)

    state = _build_state(
        fallback_context={
            "trigger": "final_output_unreachable",
            "final_step_id": "s3",
            "failed_steps": [{"step_id": "s1", "node_name": "order_agent", "status": "failed", "error": "boom"}],
            "partial_results": [{"step_id": "s2", "node_name": "product_agent", "text": "done"}],
            "reason_text": "最终步骤不可达",
        }
    )
    result = chat_module.chat_agent(state)

    assert result["results"]["chat"]["mode"] == "fallback"
    assert "final_output_unreachable" in model.messages[1].content


def test_chat_agent_keeps_normal_chat_mode_without_fallback_context(monkeypatch):
    model = _DummyModel()
    monkeypatch.setattr(chat_module, "create_chat_model", lambda *args, **kwargs: model)

    result = chat_module.chat_agent(_build_state())
    assert result["results"]["chat"]["mode"] == "chat"
    assert model.messages[1].content == "你好"


def test_chat_agent_uses_history_messages_when_present(monkeypatch):
    model = _DummyModel()
    monkeypatch.setattr(chat_module, "create_chat_model", lambda *args, **kwargs: model)

    history_messages = [
        HumanMessage(content="上一轮问题"),
        AIMessage(content="上一轮回答"),
        HumanMessage(content="本轮问题"),
    ]
    result = chat_module.chat_agent(_build_state(history_messages=history_messages))

    assert result["results"]["chat"]["mode"] == "chat"
    assert len(model.messages) == 4
    assert isinstance(model.messages[1], HumanMessage)
    assert isinstance(model.messages[2], AIMessage)
    assert isinstance(model.messages[3], HumanMessage)
    assert model.messages[3].content == "本轮问题"
