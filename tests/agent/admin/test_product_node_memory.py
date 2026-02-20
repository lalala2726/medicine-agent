import json

from langchain_core.messages import AIMessage, HumanMessage

import app.agent.admin.node.product_node as product_module


def _build_state(mode: str, *, directive: str = "") -> dict:
    return {
        "user_input": "帮我看这个商品库存",
        "messages": [
            HumanMessage(content="帮我查商品2001"),
            AIMessage(content="好的，我来查询"),
            HumanMessage(content="顺便看下是否上架"),
        ],
        "context": {"extracted_product_ids": ["2001"]},
        "routing": {
            "mode": mode,
            "directive": directive,
        },
    }


def test_product_instruction_includes_chat_history_in_fast_lane():
    payload = json.loads(product_module._build_instruction(_build_state("fast_lane")))
    assert payload["execution_mode"] == "fast_lane"
    assert "chat_history" in payload
    assert len(payload["chat_history"]) == 3
    assert payload["chat_history"][0]["role"] == "user"
    assert payload["chat_history"][1]["role"] == "assistant"


def test_product_instruction_uses_directive_without_chat_history_in_supervisor_loop():
    payload = json.loads(
        product_module._build_instruction(
            _build_state("supervisor_loop", directive="仅核验商品2001库存并返回上下架建议")
        )
    )
    assert payload["execution_mode"] == "supervisor_loop"
    assert payload["directive"] == "仅核验商品2001库存并返回上下架建议"
    assert "chat_history" not in payload
