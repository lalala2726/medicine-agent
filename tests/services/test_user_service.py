import asyncio

import httpx
import pytest

from app.services import user_service


def test_get_user_info_success(monkeypatch, capsys):
    """测试获取用户信息 - 成功"""
    fake_response = httpx.Response(200, json={"user_id": 1, "name": "测试用户"})

    class FakeHttpClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url):
            print(f"[DEBUG] HttpClient.get called with url: {url}")
            assert url == "/agent/user"
            return fake_response

    monkeypatch.setattr(user_service, "HttpClient", FakeHttpClient)

    result = asyncio.run(user_service.get_user_info())

    captured = capsys.readouterr()
    print(f"[RESULT] get_user_info output:\n{captured.out}")
    print(f"[RESULT] get_user_info response: status={result.status_code}, body={result.text}")

    assert result.status_code == 200
    assert "[DEBUG] HttpClient.get called with url: /agent/user" in captured.out


def test_get_user_info_not_found(monkeypatch, capsys):
    """测试获取用户信息 - 用户不存在"""
    fake_response = httpx.Response(404, json={"error": "用户不存在"})

    class FakeHttpClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url):
            print(f"[DEBUG] HttpClient.get called with url: {url}")
            return fake_response

    monkeypatch.setattr(user_service, "HttpClient", FakeHttpClient)

    result = asyncio.run(user_service.get_user_info())

    captured = capsys.readouterr()
    print(f"[RESULT] get_user_info output:\n{captured.out}")
    print(f"[RESULT] get_user_info response: status={result.status_code}")

    assert result.status_code == 404


def test_get_user_info_network_error(monkeypatch, capsys):
    """测试获取用户信息 - 网络异常"""
    class FakeHttpClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url):
            print(f"[DEBUG] HttpClient.get called with url: {url}")
            raise httpx.ConnectError("连接失败")

    monkeypatch.setattr(user_service, "HttpClient", FakeHttpClient)

    with pytest.raises(httpx.ConnectError):
        asyncio.run(user_service.get_user_info())

    captured = capsys.readouterr()
    print(f"[RESULT] get_user_info output:\n{captured.out}")


@pytest.mark.integration
def test_get_user_info_real_request(capsys):
    """集成测试: 真实请求远程服务"""
    result = asyncio.run(user_service.get_user_info())
    captured = capsys.readouterr()
    print(f"[RESULT] get_user_info (real request):\n{captured.out}")
    print(f"[RESULT] response: status={result.status_code}, body={result.text}")
