from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace

import pytest

import app.agent.assistant.tools.order_tools as order_tools

order_sub_agent = importlib.import_module("app.agent.assistant.sub_agents.order_sub_agent")


def _install_http_mocks(monkeypatch: pytest.MonkeyPatch, *, parsed_result: str):
    calls: list[dict] = []

    class FakeResponse:
        def __init__(self, data):
            self._data = data
            self.status_code = 200

        def json(self):
            return {
                "code": 200,
                "message": "ok",
                "data": self._data,
            }

    class FakeHttpClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, url: str, params=None, **kwargs):
            calls.append({"url": url, "params": params, **kwargs})
            return FakeResponse(parsed_result)

    monkeypatch.setattr(order_tools, "HttpClient", FakeHttpClient)
    return calls


def test_get_order_list_maps_query_params_to_backend_params(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = "code: 200\nmessage: ok\ndata:\n  rows: []\n  total: 0\n"
    calls = _install_http_mocks(monkeypatch, parsed_result=expected)

    result = asyncio.run(
        order_tools.get_order_list.ainvoke(
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
            "url": "/agent/admin/order/list",
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
    expected = "code: 200\nmessage: ok\ndata:\n  rows:\n  - id: O202601010001\n"
    calls = _install_http_mocks(monkeypatch, parsed_result=expected)

    result = asyncio.run(
        order_tools.get_orders_detail.ainvoke({"order_id": ["O202601010001", "O202601010002"]})
    )

    assert result == expected
    assert calls == [
        {
            "url": "/agent/admin/order/O202601010001,O202601010002",
            "params": None,
        }
    ]


@pytest.mark.parametrize(
    ("tool_obj", "expected_url"),
    [
        (order_tools.get_order_timeline, "/agent/admin/order/timeline/88"),
        (order_tools.get_order_shipping, "/agent/admin/order/shipping/88"),
    ],
)
def test_order_timeline_and_shipping_use_expected_path(
        monkeypatch: pytest.MonkeyPatch,
        tool_obj,
        expected_url: str,
) -> None:
    expected = "code: 200\nmessage: ok\ndata:\n  ok: true\n"
    calls = _install_http_mocks(monkeypatch, parsed_result=expected)

    result = asyncio.run(tool_obj.ainvoke({"order_id": 88}))

    assert result == expected
    assert calls == [
        {
            "url": expected_url,
            "params": None,
        }
    ]


def test_order_sub_agent_builds_expected_tools_and_returns_agent_output(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}
    fake_agent = object()
    fake_llm = object()

    def _fake_create_chat_model(**kwargs):
        captured["create_chat_model_kwargs"] = kwargs
        return fake_llm

    def _fake_create_agent(**kwargs):
        captured["create_agent_kwargs"] = kwargs
        return fake_agent

    def _fake_agent_invoke(agent, history_messages):
        captured["agent"] = agent
        captured["history_messages"] = history_messages
        return SimpleNamespace(content="订单子代理结果")

    monkeypatch.setattr(order_sub_agent, "create_chat_model", _fake_create_chat_model)
    monkeypatch.setattr(order_sub_agent, "create_agent", _fake_create_agent)
    monkeypatch.setattr(order_sub_agent, "agent_invoke", _fake_agent_invoke)

    result = order_sub_agent.order_sub_agent.invoke({"task_description": "  查询订单发货记录  "})

    assert result == "订单子代理结果"
    assert captured["agent"] is fake_agent
    assert captured["history_messages"] == "查询订单发货记录"

    assert captured["create_chat_model_kwargs"] == {
        "temperature": 1.0,
    }

    create_agent_kwargs = captured["create_agent_kwargs"]
    assert create_agent_kwargs["model"] is fake_llm
    assert create_agent_kwargs["tools"] == [
        order_tools.get_order_list,
        order_tools.get_orders_detail,
        order_tools.get_order_timeline,
        order_tools.get_order_shipping,
    ]
