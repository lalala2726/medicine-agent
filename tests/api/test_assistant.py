import json

from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from app.api.routes import admin_assistant as assistant_module
import app.main as main_module
from app.main import app
from app.schemas.auth import AuthUser


def _extract_payloads(response_text: str) -> list[dict]:
    lines = [line for line in response_text.splitlines() if line.startswith("data: ")]
    return [json.loads(line[len("data: "):]) for line in lines]


def _build_streaming_response(text: str) -> StreamingResponse:
    async def _stream():
        yield (
            "data: "
            + json.dumps(
                {
                    "content": {"text": text},
                    "type": "answer",
                    "is_end": False,
                    "timestamp": 1,
                },
                ensure_ascii=False,
            )
            + "\n\n"
        )
        yield (
            "data: "
            + json.dumps(
                {
                    "content": {"text": ""},
                    "type": "answer",
                    "is_end": True,
                    "timestamp": 2,
                },
                ensure_ascii=False,
            )
            + "\n\n"
        )

    return StreamingResponse(_stream(), media_type="text/event-stream")


def _auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-token"}


def _mock_auth(monkeypatch) -> None:
    async def _fake_fetch_current_user() -> AuthUser:
        return AuthUser(id=1, username="tester")

    monkeypatch.setattr(
        main_module,
        "verify_authorization",
        _fake_fetch_current_user,
    )


def test_assistant_route_delegates_to_service(monkeypatch):
    captured: dict = {}
    _mock_auth(monkeypatch)

    def _fake_assistant_chat(*, question: str, conversation_uuid: str | None = None):
        captured["question"] = question
        captured["conversation_uuid"] = conversation_uuid
        return _build_streaming_response("delegated")

    monkeypatch.setattr(assistant_module, "assistant_chat", _fake_assistant_chat)
    client = TestClient(app)

    response = client.post(
        "/admin/assistant/chat",
        headers=_auth_headers(),
        json={"question": "代理测试", "conversation_uuid": "conv-1"},
    )

    assert response.status_code == 200
    payloads = _extract_payloads(response.text)
    assert payloads[0]["content"]["text"] == "delegated"
    assert payloads[1]["is_end"] is True
    assert captured == {
        "question": "代理测试",
        "conversation_uuid": "conv-1",
    }


def test_assistant_request_defaults_conversation_uuid_to_none(monkeypatch):
    captured: dict = {}
    _mock_auth(monkeypatch)

    def _fake_assistant_chat(*, question: str, conversation_uuid: str | None = None):
        captured["question"] = question
        captured["conversation_uuid"] = conversation_uuid
        return _build_streaming_response("ok")

    monkeypatch.setattr(assistant_module, "assistant_chat", _fake_assistant_chat)
    client = TestClient(app)

    response = client.post(
        "/admin/assistant/chat",
        headers=_auth_headers(),
        json={"question": "hi"},
    )

    assert response.status_code == 200
    assert captured["question"] == "hi"
    assert captured["conversation_uuid"] is None


def test_assistant_request_rejects_legacy_conversion_uuid(monkeypatch):
    called = {"value": False}
    _mock_auth(monkeypatch)

    def _fake_assistant_chat(*, question: str, conversation_uuid: str | None = None):
        called["value"] = True
        return _build_streaming_response("should-not-run")

    monkeypatch.setattr(assistant_module, "assistant_chat", _fake_assistant_chat)
    client = TestClient(app)

    response = client.post(
        "/admin/assistant/chat",
        headers=_auth_headers(),
        json={"question": "hi", "conversion_uuid": "legacy"},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == 400
    assert body["message"] == "Validation Failed"
    assert any(
        item["field"] == "conversion_uuid" and item["type"] == "extra_forbidden"
        for item in body["errors"]
    )
    assert called["value"] is False
