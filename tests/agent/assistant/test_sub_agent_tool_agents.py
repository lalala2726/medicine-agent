from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.agent.assistant.tools import analytics_tool, order_tool, product_tool


@pytest.mark.parametrize(
    ("module", "tool_obj", "response_text"),
    [
        (order_tool, order_tool.order_tool_agent, "订单子代理结果"),
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
    monkeypatch.setattr(module, "create_agent_instance", lambda **_kwargs: fake_agent)
    monkeypatch.setattr(
        module,
        "agent_invoke",
        lambda _agent, _input_messages: {"messages": [AIMessage(content=response_text)]},
    )

    runtime = SimpleNamespace(tool_call_id="call_sub_agent_test")
    result = tool_obj.func("测试任务", runtime=runtime)

    assert isinstance(result, ToolMessage)
    assert result.content == response_text
    assert result.tool_call_id == "call_sub_agent_test"
    assert isinstance(result.artifact, dict)
    trace = result.artifact.get("agent_trace")
    assert isinstance(trace, dict)
    for key in (
            "text",
            "model_name",
            "usage",
            "is_usage_complete",
            "tool_calls",
            "raw_content",
    ):
        assert key in trace


def test_order_sub_agent_returns_fallback_message_when_result_empty(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_agent = object()
    monkeypatch.setattr(order_tool, "create_agent_instance", lambda **_kwargs: fake_agent)
    monkeypatch.setattr(order_tool, "agent_invoke", lambda _agent, _input_messages: {"messages": []})

    runtime = SimpleNamespace(tool_call_id="call_sub_agent_empty")
    result = order_tool.order_tool_agent.func("测试任务", runtime=runtime)

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call_sub_agent_empty"
    assert "未获取到订单数据" in str(result.content)
