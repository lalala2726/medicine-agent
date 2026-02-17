from langchain_core.messages import AIMessage

import app.agent.admin.node.summary_node as summary_module


class _DummyResponse:
    def __init__(self, content: str):
        self.content = content


class _DummyModel:
    def __init__(self):
        self.captured_messages = None

    def invoke(self, messages):
        self.captured_messages = messages
        return _DummyResponse("summary done")


def _build_step(
        *,
        include_user_input: bool = False,
        include_chat_history: bool = False,
        final_output: bool = True,
) -> dict:
    return {
        "step_id": "s3",
        "node_name": "summary_agent",
        "task_description": "汇总输出",
        "required_depends_on": ["s1", "s2"],
        "optional_depends_on": [],
        "read_from": ["s1", "s2"],
        "include_user_input": include_user_input,
        "include_chat_history": include_chat_history,
        "final_output": final_output,
    }


def _build_state(step: dict) -> dict:
    return {
        "user_input": "请总结",
        "user_intent": {},
        "plan": [step],
        "routing": {
            "route_target": "coordinator_agent",
            "next_nodes": ["summary_agent"],
            "current_step_map": {"summary_agent": step},
        },
        "order_context": {},
        "product_context": {},
        "aftersale_context": {},
        "excel_context": {},
        "history_messages": [AIMessage(content="上次回答")],
        "step_outputs": {
            "s1": {
                "step_id": "s1",
                "node_name": "order_agent",
                "status": "completed",
                "text": "order done",
                "output": {"order": 1},
            },
            "s2": {
                "step_id": "s2",
                "node_name": "product_agent",
                "status": "completed",
                "text": "product done",
                "output": {"product": 2},
            },
        },
        "shared_memory": {},
        "results": {},
        "errors": [],
    }


def test_summary_agent_default_does_not_include_user_or_history(monkeypatch):
    model = _DummyModel()
    monkeypatch.setattr(summary_module, "create_chat_model", lambda **_kwargs: model)

    step = _build_step(include_user_input=False, include_chat_history=False)
    result = summary_module.summary_agent(_build_state(step))

    assert result["results"]["summary"]["content"] == "summary done"
    assert result["step_outputs"]["s3"]["status"] == "completed"
    user_prompt = model.captured_messages[1].content
    assert "history_messages" not in user_prompt
    assert "请总结" not in user_prompt


def test_summary_agent_can_include_user_and_history(monkeypatch):
    model = _DummyModel()
    monkeypatch.setattr(summary_module, "create_chat_model", lambda **_kwargs: model)

    step = _build_step(include_user_input=True, include_chat_history=True)
    summary_module.summary_agent(_build_state(step))

    user_prompt = model.captured_messages[1].content
    assert "请总结" in user_prompt
    assert "history_messages" in user_prompt


def test_summary_agent_writes_failed_step_output_on_error(monkeypatch):
    class _BoomModel:
        def invoke(self, _messages):
            raise RuntimeError("boom")

    monkeypatch.setattr(summary_module, "create_chat_model", lambda **_kwargs: _BoomModel())
    step = _build_step()
    result = summary_module.summary_agent(_build_state(step))

    assert result["results"]["summary"]["content"] == "总结节点暂时不可用，请稍后重试。"
    assert result["step_outputs"]["s3"]["status"] == "failed"
