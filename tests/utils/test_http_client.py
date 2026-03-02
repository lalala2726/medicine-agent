import asyncio

import httpx
import pytest

import app.utils.http_client as http_client_module
from app.core.exception.exceptions import ServiceException
from app.utils.http_client import HttpClient


class _DummyLogger:
    def __init__(self):
        self.info_logs: list[str] = []
        self.warning_logs: list[str] = []
        self.error_logs: list[str] = []

    @staticmethod
    def _render(message, *args):
        if not args:
            return message
        try:
            return message.format(*args)
        except Exception:
            try:
                return message % args
            except Exception:
                return f"{message} {args}"

    def info(self, message, *args):
        self.info_logs.append(self._render(message, *args))

    def warning(self, message, *args):
        self.warning_logs.append(self._render(message, *args))

    def error(self, message, *args):
        self.error_logs.append(self._render(message, *args))


def _build_response(
        method: str,
        url: str,
        status_code: int = 200,
        body: str = "ok",
        json_body: dict | None = None,
) -> httpx.Response:
    request = httpx.Request(method, url)
    if json_body is not None:
        return httpx.Response(status_code=status_code, request=request, json=json_body)
    return httpx.Response(status_code=status_code, request=request, text=body)


def test_http_client_logs_request_when_enabled(monkeypatch):
    monkeypatch.setenv("HTTP_CLIENT_LOG_ENABLED", "true")
    client = HttpClient(base_url="http://example.com")

    dummy_logger = _DummyLogger()
    monkeypatch.setattr(http_client_module, "logger", dummy_logger)

    async def _fake_request(**kwargs):
        return _build_response(kwargs["method"], str(kwargs["url"]))

    monkeypatch.setattr(client._client, "request", _fake_request)

    asyncio.run(client.get("/ping", headers={"Authorization": "Bearer test"}))
    asyncio.run(client.close())

    assert any("HTTP request:" in item for item in dummy_logger.info_logs)
    assert any("HTTP response:" in item for item in dummy_logger.info_logs)


def test_http_client_log_switch_takes_effect_without_recreate(monkeypatch):
    monkeypatch.setenv("HTTP_CLIENT_LOG_ENABLED", "false")
    client = HttpClient(base_url="http://example.com")

    dummy_logger = _DummyLogger()
    monkeypatch.setattr(http_client_module, "logger", dummy_logger)

    async def _fake_request(**kwargs):
        return _build_response(kwargs["method"], str(kwargs["url"]))

    monkeypatch.setattr(client._client, "request", _fake_request)

    asyncio.run(client.get("/first", headers={"Authorization": "Bearer test"}))
    assert dummy_logger.info_logs == []

    monkeypatch.setenv("HTTP_CLIENT_LOG_ENABLED", "true")
    asyncio.run(client.get("/second", headers={"Authorization": "Bearer test"}))
    asyncio.run(client.close())

    assert any("HTTP request:" in item for item in dummy_logger.info_logs)


def test_http_client_logs_block_reason_when_auth_missing(monkeypatch):
    monkeypatch.setenv("HTTP_CLIENT_LOG_ENABLED", "true")
    client = HttpClient(base_url="http://example.com")

    dummy_logger = _DummyLogger()
    monkeypatch.setattr(http_client_module, "logger", dummy_logger)

    with pytest.raises(Exception):
        asyncio.run(client.get("/no-auth"))

    asyncio.run(client.close())
    assert any("HTTP request blocked before send:" in item for item in dummy_logger.warning_logs)


def test_http_client_keeps_raw_response_as_default(monkeypatch):
    client = HttpClient(base_url="http://example.com")

    async def _fake_request(**kwargs):
        return _build_response(kwargs["method"], str(kwargs["url"]), body="raw-body")

    monkeypatch.setattr(client._client, "request", _fake_request)

    result = asyncio.run(client.get("/raw", headers={"Authorization": "Bearer test"}))
    asyncio.run(client.close())

    assert isinstance(result, httpx.Response)
    assert result.text == "raw-body"


def test_http_client_returns_json_data_when_response_format_json(monkeypatch):
    client = HttpClient(base_url="http://example.com")

    async def _fake_request(**kwargs):
        return _build_response(
            kwargs["method"],
            str(kwargs["url"]),
            json_body={"code": 200, "message": "ok", "data": {"id": 1, "name": "alice"}},
        )

    monkeypatch.setattr(client._client, "request", _fake_request)

    result = asyncio.run(
        client.get(
            "/json-data",
            headers={"Authorization": "Bearer test"},
            response_format="json",
        )
    )
    asyncio.run(client.close())

    assert result == {"id": 1, "name": "alice"}


def test_http_client_returns_json_envelope_when_enabled(monkeypatch):
    client = HttpClient(base_url="http://example.com")

    async def _fake_request(**kwargs):
        return _build_response(
            kwargs["method"],
            str(kwargs["url"]),
            json_body={
                "code": 200,
                "message": "ok",
                "data": {"rows": [1, 2]},
                "timestamp": 1730000000000,
            },
        )

    monkeypatch.setattr(client._client, "request", _fake_request)

    result = asyncio.run(
        client.get(
            "/json-envelope",
            headers={"Authorization": "Bearer test"},
            response_format="json",
            include_envelope=True,
        )
    )
    asyncio.run(client.close())

    assert result == {
        "code": 200,
        "message": "ok",
        "data": {"rows": [1, 2]},
        "timestamp": 1730000000000,
    }


def test_http_client_raises_service_exception_for_non_success_business_code(monkeypatch):
    client = HttpClient(base_url="http://example.com")

    async def _fake_request(**kwargs):
        return _build_response(
            kwargs["method"],
            str(kwargs["url"]),
            json_body={"code": 500, "message": "biz error", "data": None},
        )

    monkeypatch.setattr(client._client, "request", _fake_request)

    with pytest.raises(ServiceException) as exc_info:
        asyncio.run(
            client.get(
                "/biz-error",
                headers={"Authorization": "Bearer test"},
                response_format="json",
                include_envelope=True,
            )
        )

    asyncio.run(client.close())
    assert exc_info.value.code == 500
    assert exc_info.value.message == "biz error"
