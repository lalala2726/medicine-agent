from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import app.agent.assistant.tools.user_tool as user_tool


def _install_http_mocks(monkeypatch: pytest.MonkeyPatch, *, parsed_result: str):
    calls: list[dict] = []

    class FakeHttpClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, url: str, params=None, **kwargs):
            calls.append({"url": url, "params": params, **kwargs})
            return parsed_result

    monkeypatch.setattr(user_tool, "HttpClient", FakeHttpClient)
    return calls


def test_get_admin_user_list_maps_snake_case_to_backend_params(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = "code: 200\nmessage: ok\ndata:\n  rows: []\n  total: 0\n"
    calls = _install_http_mocks(monkeypatch, parsed_result=expected)

    result = asyncio.run(
        user_tool.get_admin_user_list.ainvoke(
            {
                "page_num": 2,
                "page_size": 50,
                "id": 1001,
                "username": "alice",
                "nickname": "Alice",
                "avatar": "https://example.com/avatar.png",
                "roles": "admin",
                "status": 1,
                "create_by": "system",
            }
        )
    )

    assert result == expected
    assert calls == [
        {
            "url": "/agent/admin/user/list",
            "params": {
                "pageNum": 2,
                "pageSize": 50,
                "id": 1001,
                "username": "alice",
                "nickname": "Alice",
                "avatar": "https://example.com/avatar.png",
                "roles": "admin",
                "status": 1,
                "createBy": "system",
            },
            "response_format": "yaml",
            "include_envelope": True,
        }
    ]


@pytest.mark.parametrize(
    ("tool_obj", "expected_url"),
    [
        (user_tool.get_admin_user_detail, "/agent/admin/user/88/detail"),
        (user_tool.get_admin_user_wallet, "/agent/admin/user/88/wallet"),
    ],
)
def test_user_detail_and_wallet_use_expected_path(
        monkeypatch: pytest.MonkeyPatch,
        tool_obj,
        expected_url: str,
) -> None:
    expected = "code: 200\nmessage: ok\ndata:\n  ok: true\n"
    calls = _install_http_mocks(monkeypatch, parsed_result=expected)

    result = asyncio.run(tool_obj.ainvoke({"user_id": 88}))

    assert result == expected
    assert calls == [
        {
            "url": expected_url,
            "params": None,
            "response_format": "yaml",
            "include_envelope": True,
        }
    ]


@pytest.mark.parametrize(
    ("tool_obj", "expected_url"),
    [
        (user_tool.get_admin_user_wallet_flow, "/agent/admin/user/99/wallet_flow"),
        (user_tool.get_admin_user_consume_info, "/agent/admin/user/99/consume_info"),
    ],
)
def test_wallet_flow_and_consume_info_use_expected_path_and_pagination(
        monkeypatch: pytest.MonkeyPatch,
        tool_obj,
        expected_url: str,
) -> None:
    expected = "code: 200\nmessage: ok\ndata:\n  rows: []\n  total: 0\n"
    calls = _install_http_mocks(monkeypatch, parsed_result=expected)

    result = asyncio.run(tool_obj.ainvoke({"user_id": 99, "page_num": 3, "page_size": 20}))

    assert result == expected
    assert calls == [
        {
            "url": expected_url,
            "params": {
                "pageNum": 3,
                "pageSize": 20,
            },
            "response_format": "yaml",
            "include_envelope": True,
        }
    ]


def test_user_tool_agent_builds_expected_tools_and_returns_agent_output(
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
        return SimpleNamespace(content="用户子代理结果")

    monkeypatch.setattr(user_tool, "create_agent", _fake_create_agent)
    monkeypatch.setattr(user_tool, "agent_invoke", _fake_agent_invoke)

    result = user_tool.user_tool_agent.invoke({"task_description": "  查询用户列表  "})

    assert result == "用户子代理结果"
    assert captured["agent"] is fake_agent
    assert captured["history_messages"] == "查询用户列表"

    create_agent_kwargs = captured["create_agent_kwargs"]
    assert create_agent_kwargs["model"] == "qwen-flash"
    assert create_agent_kwargs["llm_kwargs"] == {"temperature": 0.2}
    assert create_agent_kwargs["tools"] == [
        user_tool.get_admin_user_list,
        user_tool.get_admin_user_detail,
        user_tool.get_admin_user_wallet,
        user_tool.get_admin_user_wallet_flow,
        user_tool.get_admin_user_consume_info,
    ]
