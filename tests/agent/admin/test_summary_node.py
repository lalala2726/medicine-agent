import json

import pytest
from langchain_core.messages import HumanMessage

import app.agent.admin.node.summary_node as summary_module
from app.agent.admin.model_policy import DEFAULT_NODE_GOAL


class _DummyResponse:
    def __init__(self, content: str):
        self.content = content


def _build_state(*, node_goal: str | None = None) -> dict:
    routing = {
        "mode": "supervisor_loop",
        "finished": True,
        "task_difficulty": "normal",
    }
    if node_goal is not None:
        routing["node_goal"] = node_goal

    return {
        "user_input": "我需要前5个订单对应商品的说明书结论",
        "routing": routing,
        "messages": [HumanMessage(content="请继续查询")],
        "context": {
            "agent_outputs": {
                "order_agent": {
                    "status": "completed",
                    "content": "已获取订单与商品ID映射",
                },
                "product_agent": {
                    "status": "completed",
                    "content": "已获取商品说明书摘要",
                },
            },
            "handoff_ids": {"order_ids": ["O1", "O2"], "product_ids": ["P1", "P2"]},
        },
        "results": {
            "order": {"content": "订单数据"},
            "product": {"content": "说明书数据"},
        },
        "execution_traces": [],
        "errors": [],
    }


def test_summary_agent_outputs_final_summary(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class _Model:
        def invoke(self, messages):
            captured["messages"] = messages
            return _DummyResponse("结论：已找到2个订单对应商品说明书。")

    monkeypatch.setattr(summary_module, "create_chat_model", lambda **_kwargs: _Model())

    result = summary_module.summary_agent(
        _build_state(node_goal="围绕说明书先给结论，再按订单列明细。")
    )

    assert result["results"]["summary"]["mode"] == "summary"
    assert result["results"]["summary"]["is_end"] is True
    assert result["results"]["summary"]["content"] == "结论：已找到2个订单对应商品说明书。"
    assert result["context"]["last_agent"] == "summary_agent"
    assert result["context"]["last_agent_response"] == "结论：已找到2个订单对应商品说明书。"

    input_messages = captured["messages"]
    assert isinstance(input_messages, list)
    payload = json.loads(str(input_messages[1].content))
    assert payload["node_goal"] == "围绕说明书先给结论，再按订单列明细。"
    assert payload["context"]["agent_outputs"]["order_agent"]["status"] == "completed"
    assert payload["context"]["shared_context"]["handoff_ids"]["order_ids"] == ["O1", "O2"]


def test_summary_agent_reports_unable_when_model_fails(monkeypatch: pytest.MonkeyPatch):
    def _raise_model(**_kwargs):
        raise RuntimeError("llm unavailable")

    monkeypatch.setattr(summary_module, "create_chat_model", _raise_model)

    result = summary_module.summary_agent(_build_state(node_goal="按订单说明书汇总"))
    assert result["results"]["summary"]["mode"] == "summary"
    assert result["results"]["summary"]["is_end"] is True
    assert "无法完成" in result["results"]["summary"]["content"]


def test_summary_agent_uses_default_goal_when_missing(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class _Model:
        def invoke(self, messages):
            captured["messages"] = messages
            return _DummyResponse("已按默认规则整理输出。")

    monkeypatch.setattr(summary_module, "create_chat_model", lambda **_kwargs: _Model())

    result = summary_module.summary_agent(_build_state(node_goal=None))
    assert result["results"]["summary"]["content"] == "已按默认规则整理输出。"

    input_messages = captured["messages"]
    assert isinstance(input_messages, list)
    payload = json.loads(str(input_messages[1].content))
    assert payload["node_goal"] == DEFAULT_NODE_GOAL
