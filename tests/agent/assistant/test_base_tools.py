from __future__ import annotations

import asyncio

import pytest

import app.agent.assistant.tools.base_tools as base_tools


def test_get_user_info_requests_yaml_envelope(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []
    expected = "code: 200\nmessage: ok\ndata:\n  id: 1\n"

    class FakeHttpClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, url: str, params=None, **kwargs):
            calls.append({"url": url, "params": params, **kwargs})
            return expected

    monkeypatch.setattr(base_tools, "HttpClient", FakeHttpClient)

    result = asyncio.run(base_tools.get_user_info.ainvoke({}))

    assert result == expected
    assert calls == [
        {
            "url": "/agent/info",
            "params": None,
            "response_format": "yaml",
            "include_envelope": True,
        }
    ]
