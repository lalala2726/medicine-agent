from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

import app.agent.admin.agent_utils as admin_utils
from app.agent.admin.agent_utils import NodeExecutionResult


def _build_state(mode: str, *, directive: str = "") -> dict:
    return {
        "user_input": "帮我查一下",
        "messages": [
            HumanMessage(content="上次订单是什么"),
            AIMessage(content="请提供订单号"),
            HumanMessage(content="订单号123"),
        ],
        "context": {"extracted_order_ids": ["123"]},
        "routing": {
            "mode": mode,
            "directive": directive,
        },
        "results": {},
        "execution_traces": [],
        "errors": [],
    }


def test_build_mode_aware_instruction_payload_fast_lane_includes_history():
    payload = admin_utils.build_mode_aware_instruction_payload(_build_state("fast_lane"))

    assert payload["execution_mode"] == "fast_lane"
    assert "chat_history" in payload
    assert len(payload["chat_history"]) == 3
    assert payload["chat_history"][0]["role"] == "user"


def test_build_mode_aware_instruction_payload_supervisor_loop_uses_directive():
    payload = admin_utils.build_mode_aware_instruction_payload(
        _build_state("supervisor_loop", directive="仅查询订单123状态")
    )

    assert payload["execution_mode"] == "supervisor_loop"
    assert payload["directive"] == "仅查询订单123状态"
    assert "chat_history" not in payload


def test_run_standard_tool_worker_returns_standard_structure(monkeypatch):
    class _DummyLLM:
        pass

    monkeypatch.setattr(admin_utils, "create_chat_model", lambda **_kwargs: _DummyLLM())

    def _fake_execute_tool_node(**_kwargs):
        return NodeExecutionResult(
            content="已查到订单123",
            status="completed",
            error=None,
            model_name="qwen3-max",
            input_messages=[],
            tool_calls=[{"tool_output": {"orderId": "123"}}],
            diagnostics={},
            stream_chunks=[],
        )

    monkeypatch.setattr(admin_utils, "execute_tool_node", _fake_execute_tool_node)

    update = admin_utils.run_standard_tool_worker(
        state=_build_state("fast_lane"),
        node_name="order_agent",
        result_key="order",
        system_prompt="你是订单助手",
        tools=[],
        fallback_content="fallback",
        fallback_error="error",
    )

    assert update["results"]["order"]["content"] == "已查到订单123"
    assert update["results"]["order"]["is_end"] is True
    assert update["context"]["last_agent"] == "order_agent"
    assert update["context"]["extracted_order_ids"] == ["123"]
    assert update["execution_traces"][0]["node_name"] == "order_agent"
    assert update["messages"][0].content == "已查到订单123"
