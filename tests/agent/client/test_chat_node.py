from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from app.agent.client import chat_node as chat_node_module


def test_chat_agent_registers_order_list_tool_and_keeps_tool_traces(monkeypatch):
    captured: dict = {}

    monkeypatch.setattr(
        chat_node_module,
        "create_agent_chat_llm",
        lambda **_kwargs: SimpleNamespace(model_name="chat-model"),
    )
    monkeypatch.setattr(
        chat_node_module,
        "create_agent",
        lambda **kwargs: captured.update(kwargs) or object(),
    )
    monkeypatch.setattr(
        chat_node_module,
        "agent_stream",
        lambda *_args, **_kwargs: {"streamed_text": "好的"},
    )
    monkeypatch.setattr(
        chat_node_module,
        "record_agent_trace",
        lambda **_kwargs: {
            "text": "好的",
            "model_name": "trace-model",
            "is_usage_complete": True,
            "usage": None,
            "tool_calls": [
                {
                    "tool_name": "open_user_order_list",
                    "tool_input": {"orderStatus": "PENDING_PAYMENT"},
                }
            ],
        },
    )
    monkeypatch.setattr(
        chat_node_module,
        "append_trace_and_refresh_token_usage",
        lambda traces, trace_item: (traces + [trace_item], None),
    )

    result = chat_node_module.chat_agent(
        {
            "history_messages": [HumanMessage(content="打开我的待支付订单")],
            "execution_traces": [],
        }
    )

    assert captured["tools"] == [chat_node_module.open_user_order_list]
    assert result["execution_traces"][0]["tool_calls"] == [
        {
            "tool_name": "open_user_order_list",
            "tool_input": {"orderStatus": "PENDING_PAYMENT"},
        }
    ]
