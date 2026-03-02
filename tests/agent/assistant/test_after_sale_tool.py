from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import app.agent.assistant.tools.after_sale_tool as after_sale_tool


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

    monkeypatch.setattr(after_sale_tool, "HttpClient", FakeHttpClient)
    return calls


def test_get_admin_after_sale_list_maps_snake_case_to_backend_params(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = "code: 200\nmessage: ok\ndata:\n  rows: []\n  total: 0\n"
    calls = _install_http_mocks(monkeypatch, parsed_result=expected)

    result = asyncio.run(
        after_sale_tool.get_admin_after_sale_list.ainvoke(
            {
                "page_num": 2,
                "page_size": 30,
                "after_sale_type": "REFUND_ONLY",
                "after_sale_status": "PENDING",
                "order_no": "O20251108123456789012",
                "user_id": 1001,
                "apply_reason": "DAMAGED",
            }
        )
    )

    assert result == expected
    assert calls == [
        {
            "url": "/agent/admin/after-sale/list",
            "params": {
                "pageNum": 2,
                "pageSize": 30,
                "afterSaleType": "REFUND_ONLY",
                "afterSaleStatus": "PENDING",
                "orderNo": "O20251108123456789012",
                "userId": 1001,
                "applyReason": "DAMAGED",
            },
        }
    ]


def test_get_admin_after_sale_detail_uses_expected_path(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = "code: 200\nmessage: ok\ndata:\n  id: 30001\n"
    calls = _install_http_mocks(monkeypatch, parsed_result=expected)

    result = asyncio.run(after_sale_tool.get_admin_after_sale_detail.ainvoke({"after_sale_id": 30001}))

    assert result == expected
    assert calls == [
        {
            "url": "/agent/admin/after-sale/detail/30001",
            "params": None,
        }
    ]


def test_after_sale_tool_agent_builds_expected_tools_and_returns_agent_output(
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
        return SimpleNamespace(content="售后子代理结果")

    monkeypatch.setattr(after_sale_tool, "create_chat_model", _fake_create_chat_model)
    monkeypatch.setattr(after_sale_tool, "create_agent", _fake_create_agent)
    monkeypatch.setattr(after_sale_tool, "agent_invoke", _fake_agent_invoke)

    result = after_sale_tool.after_sale_tool_agent.invoke({"task_description": "  查询售后详情  "})

    assert result == "售后子代理结果"
    assert captured["agent"] is fake_agent
    assert captured["history_messages"] == "查询售后详情"

    assert captured["create_chat_model_kwargs"] == {
        "model": "qwen-flash",
        "provider": after_sale_tool.LlmProvider.ALIYUN,
        "temperature": 0.2,
    }

    create_agent_kwargs = captured["create_agent_kwargs"]
    assert create_agent_kwargs["model"] is fake_llm
    assert create_agent_kwargs["tools"] == [
        after_sale_tool.get_admin_after_sale_list,
        after_sale_tool.get_admin_after_sale_detail,
    ]
