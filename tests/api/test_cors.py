from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app
from app.schemas.auth import AuthUser


def _auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-token"}


def _mock_auth(monkeypatch) -> None:
    async def _fake_fetch_current_user() -> AuthUser:
        return AuthUser(id=1, username="tester")

    monkeypatch.setattr(
        main_module,
        "fetch_current_user_by_authorization",
        _fake_fetch_current_user,
    )


def test_cors_preflight_allows_localhost_origin():
    client = TestClient(app)

    response = client.options(
        "/image_parse/drug",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert response.headers["access-control-allow-credentials"] == "true"


def test_cors_simple_request_allows_127_origin(monkeypatch):
    _mock_auth(monkeypatch)
    client = TestClient(app)

    response = client.post(
        "/image_parse/drug",
        headers={"Origin": "http://127.0.0.1:5173", **_auth_headers()},
        json={"images": []},
    )

    assert response.status_code == 400
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"
    assert response.headers["access-control-allow-credentials"] == "true"


def test_cors_preflight_rejects_non_local_origin():
    client = TestClient(app)

    response = client.options(
        "/image_parse/drug",
        headers={
            "Origin": "http://evil.com",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers
