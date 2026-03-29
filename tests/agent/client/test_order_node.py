from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from app.agent.client.domain.order import node as order_node_module
from app.core.config_sync import AgentChatModelSlot


def test_order_agent_registers_order_list_tool_and_keeps_tool_traces(monkeypatch):
    captured: dict = {}
    captured_llm_kwargs: dict = {}

    monkeypatch.setattr(
        order_node_module,
        "create_agent_chat_llm",
        lambda **kwargs: captured_llm_kwargs.update(kwargs) or SimpleNamespace(model_name="order-model"),
    )
    monkeypatch.setattr(
        order_node_module,
        "create_agent",
        lambda **kwargs: captured.update(kwargs) or object(),
    )
    monkeypatch.setattr(
        order_node_module,
        "agent_stream",
        lambda *_args, **_kwargs: {"streamed_text": "已为你打开待支付订单列表"},
    )
    monkeypatch.setattr(
        order_node_module,
        "record_agent_trace",
        lambda **_kwargs: {
            "text": "已为你打开待支付订单列表",
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
        order_node_module,
        "append_trace_and_refresh_token_usage",
        lambda traces, trace_item: (traces + [trace_item], None),
    )

    result = order_node_module.order_agent(
        {
            "history_messages": [HumanMessage(content="打开我的待支付订单")],
            "execution_traces": [],
        }
    )

    assert captured["tools"] == [
        order_node_module.open_user_order_list,
        order_node_module.get_order_detail,
        order_node_module.get_order_shipping,
        order_node_module.get_order_timeline,
        order_node_module.check_order_cancelable,
    ]
    assert captured_llm_kwargs["slot"] is AgentChatModelSlot.CLIENT_ORDER
    assert result["execution_traces"][0]["node_name"] == "order_agent"
    assert result["execution_traces"][0]["tool_calls"] == [
        {
            "tool_name": "open_user_order_list",
            "tool_input": {"orderStatus": "PENDING_PAYMENT"},
        }
    ]
