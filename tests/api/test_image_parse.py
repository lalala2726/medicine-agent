from fastapi.testclient import TestClient

from app.api.routes import image_parse as image_parse_module
from app.main import app


def test_image_parse_success(monkeypatch):
    monkeypatch.setattr(image_parse_module, "parse_drug_images", lambda images: {"ok": True})
    client = TestClient(app)

    response = client.post("/api/image/parse/drug", json={"images": ["abc"]})

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 200
    assert body["message"] == "解析成功"
    assert body["data"] == {"ok": True}


def test_image_parse_requires_images():
    client = TestClient(app)

    response = client.post("/api/image/parse/drug", json={"images": []})

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == 400
    assert body["message"] == "图片不能为空"
