from fastapi.testclient import TestClient

from app.main import app


def test_cors_preflight_allows_localhost_origin():
    client = TestClient(app)

    response = client.options(
        "/api/image/parse/drug",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert response.headers["access-control-allow-credentials"] == "true"


def test_cors_simple_request_allows_127_origin():
    client = TestClient(app)

    response = client.post(
        "/api/image/parse/drug",
        headers={"Origin": "http://127.0.0.1:5173"},
        json={"images": []},
    )

    assert response.status_code == 400
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"
    assert response.headers["access-control-allow-credentials"] == "true"


def test_cors_preflight_rejects_non_local_origin():
    client = TestClient(app)

    response = client.options(
        "/api/image/parse/drug",
        headers={
            "Origin": "http://evil.com",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers
