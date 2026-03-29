from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from app.agent.client.domain.product import node as product_node_module
from app.core.config_sync import AgentChatModelSlot


def test_product_agent_registers_product_tools_and_keeps_tool_traces(monkeypatch):
    captured: dict = {}
    captured_llm_kwargs: dict = {}

    monkeypatch.setattr(
        product_node_module,
        "create_agent_chat_llm",
        lambda **kwargs: captured_llm_kwargs.update(kwargs) or SimpleNamespace(model_name="product-model"),
    )
    monkeypatch.setattr(
        product_node_module,
        "create_agent",
        lambda **kwargs: captured.update(kwargs) or object(),
    )
    monkeypatch.setattr(
        product_node_module,
        "agent_stream",
        lambda *_args, **_kwargs: {"streamed_text": "我先帮你找相关商品"},
    )
    monkeypatch.setattr(
        product_node_module,
        "record_agent_trace",
        lambda **_kwargs: {
            "text": "我先帮你找相关商品",
            "model_name": "trace-model",
            "is_usage_complete": True,
            "usage": None,
            "tool_calls": [
                {
                    "tool_name": "search_products",
                    "tool_input": {"keyword": "维生素", "page_num": 1, "page_size": 10},
                }
            ],
        },
    )
    monkeypatch.setattr(
        product_node_module,
        "append_trace_and_refresh_token_usage",
        lambda traces, trace_item: (traces + [trace_item], None),
    )

    result = product_node_module.product_agent(
        {
            "history_messages": [HumanMessage(content="帮我找点维生素")],
            "execution_traces": [],
        }
    )

    assert captured["tools"] == [
        product_node_module.search_products,
        product_node_module.get_product_detail,
        product_node_module.get_product_spec,
    ]
    assert captured_llm_kwargs["slot"] is AgentChatModelSlot.CLIENT_PRODUCT
    assert result["execution_traces"][0]["node_name"] == "product_agent"
    assert result["execution_traces"][0]["tool_calls"] == [
        {
            "tool_name": "search_products",
            "tool_input": {"keyword": "维生素", "page_num": 1, "page_size": 10},
        }
    ]
