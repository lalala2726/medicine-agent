from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from app.agent.client.domain.after_sale import node as after_sale_node_module


def test_after_sale_agent_registers_after_sale_tools_and_keeps_tool_traces(
        monkeypatch,
):
    captured: dict = {}

    monkeypatch.setattr(
        after_sale_node_module,
        "create_agent_chat_llm",
        lambda **_kwargs: SimpleNamespace(model_name="after-sale-model"),
    )
    monkeypatch.setattr(
        after_sale_node_module,
        "create_agent",
        lambda **kwargs: captured.update(kwargs) or object(),
    )
    monkeypatch.setattr(
        after_sale_node_module,
        "agent_stream",
        lambda *_args, **_kwargs: {"streamed_text": "我先帮你核对售后资格"},
    )
    monkeypatch.setattr(
        after_sale_node_module,
        "record_agent_trace",
        lambda **_kwargs: {
            "text": "我先帮你核对售后资格",
            "model_name": "trace-model",
            "is_usage_complete": True,
            "usage": None,
            "tool_calls": [
                {
                    "tool_name": "check_after_sale_eligibility",
                    "tool_input": {"order_no": "O202603160001"},
                }
            ],
        },
    )
    monkeypatch.setattr(
        after_sale_node_module,
        "append_trace_and_refresh_token_usage",
        lambda traces, trace_item: (traces + [trace_item], None),
    )

    result = after_sale_node_module.after_sale_agent(
        {
            "history_messages": [HumanMessage(content="这个订单能退款吗")],
            "execution_traces": [],
        }
    )

    assert captured["tools"] == [
        after_sale_node_module.get_after_sale_detail,
        after_sale_node_module.check_after_sale_eligibility,
        after_sale_node_module.open_user_order_list,
        after_sale_node_module.open_user_after_sale_list,
    ]
    assert result["execution_traces"][0]["node_name"] == "after_sale_agent"
    assert result["execution_traces"][0]["tool_calls"] == [
        {
            "tool_name": "check_after_sale_eligibility",
            "tool_input": {"order_no": "O202603160001"},
        }
    ]
