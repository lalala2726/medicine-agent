from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

after_sale_sub_agent_module = importlib.import_module("app.agent.assistant.sub_agents.after_sale_sub_agent")
analytics_sub_agent_module = importlib.import_module("app.agent.assistant.sub_agents.analytics_sub_agent")
order_sub_agent_module = importlib.import_module("app.agent.assistant.sub_agents.order_sub_agent")
product_sub_agent_module = importlib.import_module("app.agent.assistant.sub_agents.product_sub_agent")


@pytest.mark.parametrize(
    ("module", "tool_obj", "response_text"),
    [
        (order_sub_agent_module, order_sub_agent_module.order_sub_agent, "订单子代理结果"),
        (
                after_sale_sub_agent_module,
                after_sale_sub_agent_module.after_sale_sub_agent,
                "售后子代理结果",
        ),
        (product_sub_agent_module, product_sub_agent_module.product_sub_agent, "商品子代理结果"),
        (
                analytics_sub_agent_module,
                analytics_sub_agent_module.analytics_sub_agent,
                "分析子代理结果",
        ),
    ],
)
def test_sub_agent_returns_text_result_with_llm_agent_runtime(
        monkeypatch: pytest.MonkeyPatch,
        module,
        tool_obj,
        response_text: str,
) -> None:
    captured: dict[str, object] = {}
    fake_agent = object()
    fake_llm = object()
    monkeypatch.setattr(module, "create_chat_model", lambda **_kwargs: fake_llm)
    monkeypatch.setattr(
        module,
        "create_agent",
        lambda **kwargs: (captured.setdefault("create_agent_kwargs", kwargs), fake_agent)[1],
    )
    monkeypatch.setattr(
        module,
        "agent_invoke",
        lambda _agent, _input_messages: SimpleNamespace(content=response_text),
    )

    result = tool_obj.func("测试任务")

    assert isinstance(result, str)
    assert result == response_text
    assert captured["create_agent_kwargs"]["model"] is fake_llm


def test_order_sub_agent_returns_fallback_message_when_result_empty(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    fake_agent = object()
    fake_llm = object()
    monkeypatch.setattr(order_sub_agent_module, "create_chat_model", lambda **_kwargs: fake_llm)
    monkeypatch.setattr(
        order_sub_agent_module,
        "create_agent",
        lambda **kwargs: (captured.setdefault("create_agent_kwargs", kwargs), fake_agent)[1],
    )
    monkeypatch.setattr(
        order_sub_agent_module,
        "agent_invoke",
        lambda _agent, _input_messages: SimpleNamespace(content=""),
    )

    result = order_sub_agent_module.order_sub_agent.func("测试任务")

    assert isinstance(result, str)
    assert result == "暂无数据"
    assert captured["create_agent_kwargs"]["model"] is fake_llm
