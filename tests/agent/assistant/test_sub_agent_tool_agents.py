from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.agent.assistant.tools import after_sale_tool, analytics_tool, order_tool, product_tool


@pytest.mark.parametrize(
    ("module", "tool_obj", "response_text"),
    [
        (order_tool, order_tool.order_tool_agent, "订单子代理结果"),
        (after_sale_tool, after_sale_tool.after_sale_tool_agent, "售后子代理结果"),
        (product_tool, product_tool.product_tool_agent, "商品子代理结果"),
        (analytics_tool, analytics_tool.analytics_tool_agent, "分析子代理结果"),
    ],
)
def test_sub_agent_tool_returns_tool_message_with_trace_artifact(
        monkeypatch: pytest.MonkeyPatch,
        module,
        tool_obj,
        response_text: str,
) -> None:
    fake_agent = object()
    monkeypatch.setattr(module, "create_agent", lambda **_kwargs: fake_agent)
    monkeypatch.setattr(
        module,
        "agent_invoke",
        lambda _agent, _input_messages: SimpleNamespace(content=response_text),
    )

    result = tool_obj.func("测试任务")

    assert isinstance(result, str)
    assert result == response_text


def test_order_sub_agent_returns_fallback_message_when_result_empty(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_agent = object()
    monkeypatch.setattr(order_tool, "create_agent", lambda **_kwargs: fake_agent)
    monkeypatch.setattr(
        order_tool,
        "agent_invoke",
        lambda _agent, _input_messages: SimpleNamespace(content=""),
    )

    result = order_tool.order_tool_agent.func("测试任务")

    assert isinstance(result, str)
    assert result == "暂无数据"
