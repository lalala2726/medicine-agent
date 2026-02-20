import json

from langchain_core.messages import AIMessage, HumanMessage

import app.agent.admin.node.order_node as order_module


def _build_state(mode: str, *, directive: str = "") -> dict:
    return {
        "user_input": "帮我查一下订单情况",
        "messages": [
            HumanMessage(content="我昨天有个订单"),
            AIMessage(content="请提供订单号"),
            HumanMessage(content="订单号是123"),
        ],
        "context": {"extracted_order_ids": ["123"]},
        "routing": {
            "mode": mode,
            "directive": directive,
        },
    }


def test_order_instruction_includes_chat_history_in_fast_lane():
    payload = json.loads(order_module._build_instruction(_build_state("fast_lane")))
    assert payload["execution_mode"] == "fast_lane"
    assert "chat_history" in payload
    assert len(payload["chat_history"]) == 3
    assert payload["chat_history"][0]["role"] == "user"
    assert payload["chat_history"][1]["role"] == "assistant"


def test_order_instruction_uses_directive_without_chat_history_in_supervisor_loop():
    payload = json.loads(
        order_module._build_instruction(
            _build_state("supervisor_loop", directive="仅查询订单123的退款次数并返回结论")
        )
    )
    assert payload["execution_mode"] == "supervisor_loop"
    assert payload["directive"] == "仅查询订单123的退款次数并返回结论"
    assert "chat_history" not in payload
