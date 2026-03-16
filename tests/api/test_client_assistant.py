import importlib
import json

from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

import app.main as main_module
from app.api.routes import client_assistant as assistant_module
from app.main import app
from app.schemas.admin_assistant_history import ConversationMessageResponse
from app.schemas.auth import AuthUser
from app.schemas.document.conversation import ConversationListItem

rate_limit_module = importlib.import_module("app.core.security.rate_limit")


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
        return AuthUser(
            id=1,
            username="tester",
            roles=[],
            permissions=[],
        )

    monkeypatch.setattr(
        main_module,
        "verify_authorization",
        _fake_fetch_current_user,
    )


def _mock_rate_limit_allow(monkeypatch) -> None:
    def _fake_evaluate_rate_limit(*, scope: str, subject_key: str, rules):
        return rate_limit_module.RateLimitCheckResult(
            allowed=True,
            retry_after_seconds=0,
            limit=10,
            remaining=9,
            reset_seconds=60,
        )

    monkeypatch.setattr(rate_limit_module, "_evaluate_rate_limit", _fake_evaluate_rate_limit)


def test_client_assistant_chat_route_delegates_to_service(monkeypatch):
    captured: dict = {}
    _mock_auth(monkeypatch)
    _mock_rate_limit_allow(monkeypatch)

    monkeypatch.setattr(
        assistant_module,
        "assistant_chat",
        lambda *, question, conversation_uuid=None: (
            captured.update(
                {
                    "question": question,
                    "conversation_uuid": conversation_uuid,
                }
            ),
            _build_streaming_response("客户端回复"),
        )[-1],
    )
    client = TestClient(app)

    response = client.post(
        "/client/assistant/chat",
        headers=_auth_headers(),
        json={"question": "客户端问题", "conversation_uuid": "client-conv-1"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert captured == {
        "question": "客户端问题",
        "conversation_uuid": "client-conv-1",
    }
    payloads = _extract_payloads(response.text)
    assert payloads[0]["content"]["text"] == "客户端回复"
    assert payloads[-1]["is_end"] is True


def test_client_assistant_chat_rejects_blank_question(monkeypatch):
    _mock_auth(monkeypatch)
    _mock_rate_limit_allow(monkeypatch)
    client = TestClient(app)

    response = client.post(
        "/client/assistant/chat",
        headers=_auth_headers(),
        json={"question": "   "},
    )

    assert response.status_code == 400


def test_client_conversation_list_route_returns_page(monkeypatch):
    _mock_auth(monkeypatch)
    monkeypatch.setattr(
        assistant_module,
        "conversation_list_service",
        lambda *, page_request: (
            [ConversationListItem(conversation_uuid="client-conv-1", title="标题1")],
            1,
        ),
    )
    client = TestClient(app)

    response = client.get(
        "/client/assistant/conversation/list?page_num=1&page_size=20",
        headers=_auth_headers(),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["data"]["rows"] == [
        {"conversation_uuid": "client-conv-1", "title": "标题1"}
    ]
    assert body["data"]["total"] == 1


def test_client_history_route_returns_serialized_messages(monkeypatch):
    _mock_auth(monkeypatch)
    monkeypatch.setattr(
        assistant_module,
        "conversation_messages_service",
        lambda *, conversation_uuid, page_request: (
            [
                ConversationMessageResponse(
                    id="msg-1",
                    role="user",
                    content="你好",
                )
            ],
            1,
        ),
    )
    client = TestClient(app)

    response = client.get(
        "/client/assistant/history/client-conv-1?page_num=1&page_size=20",
        headers=_auth_headers(),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["data"]["rows"] == [
        {"id": "msg-1", "role": "user", "content": "你好"}
    ]
    assert body["data"]["total"] == 1


def test_delete_client_conversation_route_delegates_to_service(monkeypatch):
    captured: dict = {}
    _mock_auth(monkeypatch)

    def _fake_delete_conversation(*, conversation_uuid: str):
        captured["conversation_uuid"] = conversation_uuid

    monkeypatch.setattr(
        assistant_module,
        "delete_conversation_service",
        _fake_delete_conversation,
    )
    client = TestClient(app)

    response = client.delete(
        "/client/assistant/conversation/client-conv-1",
        headers=_auth_headers(),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 200
    assert body["message"] == "删除成功"
    assert body["data"] == {"conversation_uuid": "client-conv-1"}
    assert captured == {"conversation_uuid": "client-conv-1"}


def test_update_client_conversation_title_route_delegates_to_service(monkeypatch):
    captured: dict = {}
    _mock_auth(monkeypatch)

    def _fake_update_title(*, conversation_uuid: str, title: str) -> str:
        captured["conversation_uuid"] = conversation_uuid
        captured["title"] = title
        return "新标题"

    monkeypatch.setattr(
        assistant_module,
        "update_conversation_title_service",
        _fake_update_title,
    )
    client = TestClient(app)

    response = client.put(
        "/client/assistant/conversation/client-conv-1",
        headers=_auth_headers(),
        json={"title": "  新标题  "},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 200
    assert body["message"] == "修改成功"
    assert body["data"] == {
        "conversation_uuid": "client-conv-1",
        "title": "新标题",
    }
    assert captured == {"conversation_uuid": "client-conv-1", "title": "  新标题  "}


def test_delete_client_conversation_requires_auth(monkeypatch):
    called = {"value": False}

    def _fake_delete_conversation(*, conversation_uuid: str):
        called["value"] = True

    monkeypatch.setattr(
        assistant_module,
        "delete_conversation_service",
        _fake_delete_conversation,
    )
    client = TestClient(app)

    response = client.delete("/client/assistant/conversation/client-conv-1")

    assert response.status_code == 401
    body = response.json()
    assert body["code"] == 401
    assert called["value"] is False


def test_update_client_conversation_title_requires_auth(monkeypatch):
    called = {"value": False}

    def _fake_update_title(*, conversation_uuid: str, title: str) -> str:
        called["value"] = True
        return "不会执行"

    monkeypatch.setattr(
        assistant_module,
        "update_conversation_title_service",
        _fake_update_title,
    )
    client = TestClient(app)

    response = client.put(
        "/client/assistant/conversation/client-conv-1",
        json={"title": "新标题"},
    )

    assert response.status_code == 401
    body = response.json()
    assert body["code"] == 401
    assert called["value"] is False
