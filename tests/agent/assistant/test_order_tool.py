from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import app.agent.assistant.tools.order_tool as order_tool


def _install_http_mocks(monkeypatch: pytest.MonkeyPatch, *, parsed_result: dict):
    calls: list[dict] = []
    fake_response = object()

    class FakeHttpClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, url: str, params=None):
            calls.append({"url": url, "params": params})
            return fake_response

    def _fake_parse_data(response):
        assert response is fake_response
        return parsed_result

    monkeypatch.setattr(order_tool, "HttpClient", FakeHttpClient)
    monkeypatch.setattr(order_tool.HttpResponse, "parse_data", _fake_parse_data)
    return calls


def test_get_order_list_maps_query_params_to_backend_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {"rows": [], "total": 0}
    calls = _install_http_mocks(monkeypatch, parsed_result=expected)

    result = asyncio.run(
        order_tool.get_order_list.ainvoke(
            {
                "page_num": 2,
                "page_size": 20,
                "order_no": "O202601010001",
                "pay_type": "wechat",
                "order_status": "shipped",
                "delivery_type": "express",
                "receiver_name": "张三",
                "receiver_phone": "13800138000",
            }
        )
    )

    assert result == expected
    assert calls == [
        {
            "url": "/agent/order/list",
            "params": {
                "pageNum": 2,
                "pageSize": 20,
                "orderNo": "O202601010001",
                "payType": "wechat",
                "orderStatus": "shipped",
                "deliveryType": "express",
                "receiverName": "张三",
                "receiverPhone": "13800138000",
            },
        }
    ]


def test_get_orders_detail_formats_order_ids_into_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {"rows": [{"id": "O202601010001"}]}
    calls = _install_http_mocks(monkeypatch, parsed_result=expected)

    result = asyncio.run(
        order_tool.get_orders_detail.ainvoke({"order_id": ["O202601010001", "O202601010002"]})
    )

    assert result == expected
    assert calls == [{"url": "/agent/order/O202601010001,O202601010002", "params": None}]


@pytest.mark.parametrize(
    ("tool_obj", "expected_url"),
    [
        (order_tool.get_order_timeline, "/agent/order/timeline/88"),
        (order_tool.get_order_shipping, "/agent/order/shipping/88"),
    ],
)
def test_order_timeline_and_shipping_use_expected_path(
    monkeypatch: pytest.MonkeyPatch,
    tool_obj,
    expected_url: str,
) -> None:
    expected = {"ok": True}
    calls = _install_http_mocks(monkeypatch, parsed_result=expected)

    result = asyncio.run(tool_obj.ainvoke({"order_id": 88}))

    assert result == expected
    assert calls == [{"url": expected_url, "params": None}]


def test_order_tool_agent_builds_expected_tools_and_returns_agent_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}
    fake_agent = object()

    def _fake_create_agent(**kwargs):
        captured["create_agent_kwargs"] = kwargs
        return fake_agent

    def _fake_agent_invoke(agent, history_messages):
        captured["agent"] = agent
        captured["history_messages"] = history_messages
        return SimpleNamespace(content="订单子代理结果")

    monkeypatch.setattr(order_tool, "create_agent", _fake_create_agent)
    monkeypatch.setattr(order_tool, "agent_invoke", _fake_agent_invoke)

    result = order_tool.order_tool_agent.invoke({"task_description": "  查询订单发货记录  "})

    assert result == "订单子代理结果"
    assert captured["agent"] is fake_agent
    assert captured["history_messages"] == "查询订单发货记录"

    create_agent_kwargs = captured["create_agent_kwargs"]
    assert create_agent_kwargs["model"] == "qwen-flash"
    assert create_agent_kwargs["llm_kwargs"] == {"temperature": 0.2}
    assert create_agent_kwargs["tools"] == [
        order_tool.get_order_list,
        order_tool.get_orders_detail,
        order_tool.get_order_timeline,
        order_tool.get_order_shipping,
    ]
