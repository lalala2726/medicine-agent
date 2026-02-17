from fastapi.testclient import TestClient

import app.main as main_module
from app.core.exceptions import ServiceException
from app.core.request_context import get_authorization_header
from app.main import app
from app.schemas.auth import AuthUser


def _authorized_headers(token: str = "demo-token") -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_business_route_requires_authorization_header():
    client = TestClient(app)

    response = client.get("/test/test1")

    assert response.status_code == 401
    assert response.json()["code"] == 401


def test_business_route_passes_with_valid_authorization(monkeypatch):
    async def _fake_fetch_current_user() -> AuthUser:
        return AuthUser(id=100, username="admin")

    monkeypatch.setattr(
        main_module,
        "verify_authorization",
        _fake_fetch_current_user,
    )
    client = TestClient(app)

    response = client.get("/test/test1", headers=_authorized_headers("token-100"))

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 200
    assert body["data"]["user_id"] == 100
    assert body["data"]["token"] == "token-100"


def test_business_route_returns_503_when_auth_service_unavailable(monkeypatch):
    async def _fake_fetch_current_user() -> AuthUser:
        raise ServiceException(code=503, message="upstream down")

    monkeypatch.setattr(
        main_module,
        "verify_authorization",
        _fake_fetch_current_user,
    )
    client = TestClient(app)

    response = client.get("/test/test1", headers=_authorized_headers())

    assert response.status_code == 503
    body = response.json()
    assert body["code"] == 503
    assert body["message"] == "upstream down"


def test_options_request_skips_authorization(monkeypatch):
    called = {"count": 0}

    async def _fake_fetch_current_user() -> AuthUser:
        called["count"] += 1
        return AuthUser(id=1, username="demo")

    monkeypatch.setattr(
        main_module,
        "verify_authorization",
        _fake_fetch_current_user,
    )
    client = TestClient(app)

    response = client.options(
        "/image_parse/drug",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert called["count"] == 0


def test_docs_route_skips_authorization(monkeypatch):
    called = {"count": 0}

    async def _fake_fetch_current_user() -> AuthUser:
        called["count"] += 1
        return AuthUser(id=1, username="demo")

    monkeypatch.setattr(
        main_module,
        "verify_authorization",
        _fake_fetch_current_user,
    )
    client = TestClient(app)

    response = client.get("/docs")

    assert response.status_code == 200
    assert called["count"] == 0


def test_consecutive_requests_do_not_leak_user_context(monkeypatch):
    async def _fake_fetch_current_user() -> AuthUser:
        authorization = get_authorization_header()
        token = authorization.split(" ", 1)[1]
        if token == "token-a":
            return AuthUser(id=1, username="user-a")
        return AuthUser(id=2, username="user-b")

    monkeypatch.setattr(
        main_module,
        "verify_authorization",
        _fake_fetch_current_user,
    )
    client = TestClient(app)

    response_a = client.get("/test/test1", headers=_authorized_headers("token-a"))
    response_b = client.get("/test/test1", headers=_authorized_headers("token-b"))

    assert response_a.status_code == 200
    assert response_b.status_code == 200
    assert response_a.json()["data"]["user_id"] == 1
    assert response_b.json()["data"]["user_id"] == 2
