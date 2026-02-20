from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.admin.agent_utils import serialize_message, serialize_messages


def test_serialize_message_preserves_role_and_content():
    message = HumanMessage(content="你好")
    payload = serialize_message(message)

    assert payload["role"] == "human"
    assert payload["content"] == "你好"


def test_serialize_message_keeps_optional_fields():
    message = AIMessage(content="结果", additional_kwargs={"tool": "get_order_list"}, name="assistant")
    payload = serialize_message(message)

    assert payload["role"] == "ai"
    assert payload["content"] == "结果"
    assert payload["additional_kwargs"] == {"tool": "get_order_list"}
    assert payload["name"] == "assistant"


def test_serialize_messages_keeps_order():
    messages = [HumanMessage(content="第一条"), AIMessage(content="第二条")]
    payload = serialize_messages(messages)

    assert len(payload) == 2
    assert payload[0]["content"] == "第一条"
    assert payload[1]["content"] == "第二条"
