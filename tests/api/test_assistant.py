import json

from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

import app.main as main_module
from app.api.routes import admin_assistant as assistant_module
from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException
from app.core.security.role_codes import RoleCode
from app.main import app
from app.schemas.admin_assistant_history import ConversationMessageResponse, ThoughtNodeResponse
from app.schemas.auth import AuthUser
from app.schemas.document.conversation import ConversationListItem


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


def _mock_auth(
        monkeypatch,
        *,
        roles: list[str] | None = None,
        permissions: list[str] | None = None,
) -> None:
    resolved_roles = [RoleCode.SUPER_ADMIN.value] if roles is None else roles
    resolved_permissions = [] if permissions is None else permissions

    async def _fake_fetch_current_user() -> AuthUser:
        return AuthUser(
            id=1,
            username="tester",
            roles=resolved_roles,
            permissions=resolved_permissions,
        )

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


def test_assistant_request_normalizes_question_and_conversation_uuid(monkeypatch):
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
        json={"question": "  请帮我查订单  ", "conversation_uuid": "  conv-1  "},
    )

    assert response.status_code == 200
    assert captured["question"] == "请帮我查订单"
    assert captured["conversation_uuid"] == "conv-1"


def test_assistant_request_rejects_blank_question_after_trim(monkeypatch):
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
        json={"question": "   ", "conversation_uuid": "conv-1"},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == 400
    assert body["message"] == "Validation Failed"
    assert called["value"] is False


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


def test_conversation_list_route_delegates_to_service(monkeypatch):
    captured: dict = {}
    _mock_auth(monkeypatch)

    def _fake_chat_list(*, page_request):
        captured["page_num"] = page_request.page_num
        captured["page_size"] = page_request.page_size
        return ([ConversationListItem(conversation_uuid="conv-1", title="会话1")], 3)

    monkeypatch.setattr(
        assistant_module,
        "conversation_list_service",
        _fake_chat_list,
    )
    client = TestClient(app)

    response = client.get(
        "/admin/assistant/conversation/list",
        headers=_auth_headers(),
        params={"page_num": 2, "page_size": 5},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 200
    assert body["data"]["rows"] == [{"conversation_uuid": "conv-1", "title": "会话1"}]
    assert body["data"]["total"] == 3
    assert body["data"]["page_num"] == 2
    assert body["data"]["page_size"] == 5
    assert captured == {"page_num": 2, "page_size": 5}


def test_conversation_list_route_uses_default_pagination(monkeypatch):
    captured: dict = {}
    _mock_auth(monkeypatch)

    def _fake_chat_list(*, page_request):
        captured["page_num"] = page_request.page_num
        captured["page_size"] = page_request.page_size
        return ([], 0)

    monkeypatch.setattr(
        assistant_module,
        "conversation_list_service",
        _fake_chat_list,
    )
    client = TestClient(app)

    response = client.get(
        "/admin/assistant/conversation/list",
        headers=_auth_headers(),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 200
    assert body["data"]["rows"] == []
    assert body["data"]["total"] == 0
    assert body["data"]["page_num"] == 1
    assert body["data"]["page_size"] == 20
    assert captured == {"page_num": 1, "page_size": 20}


def test_conversation_messages_route_delegates_to_service(monkeypatch):
    captured: dict = {}
    _mock_auth(monkeypatch)

    def _fake_conversation_messages(*, conversation_uuid: str, page_request):
        captured["conversation_uuid"] = conversation_uuid
        captured["page_num"] = page_request.page_num
        captured["page_size"] = page_request.page_size
        return [
            ConversationMessageResponse(
                id="msg-1",
                role="user",
                content="你好",
            ),
            ConversationMessageResponse(
                id="msg-2",
                role="ai",
                content="您好",
                status="success",
                thought_chain=[
                    ThoughtNodeResponse(
                        id="node-1",
                        node="planner",
                        message="planner",
                        status="success",
                        children=[],
                    )
                ],
            ),
        ]

    monkeypatch.setattr(
        assistant_module,
        "conversation_messages_service",
        _fake_conversation_messages,
    )
    client = TestClient(app)

    response = client.get(
        "/admin/assistant/conversation/conv-1/messages",
        headers=_auth_headers(),
        params={"page_num": 1, "page_size": 50},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 200
    assert body["data"][0] == {"id": "msg-1", "role": "user", "content": "你好"}
    assert body["data"][1]["id"] == "msg-2"
    assert body["data"][1]["role"] == "ai"
    assert body["data"][1]["status"] == "success"
    assert body["data"][1]["thoughtChain"][0]["node"] == "planner"
    assert captured == {"conversation_uuid": "conv-1", "page_num": 1, "page_size": 50}


def test_conversation_messages_route_uses_default_pagination(monkeypatch):
    captured: dict = {}
    _mock_auth(monkeypatch)

    def _fake_conversation_messages(*, conversation_uuid: str, page_request):
        captured["conversation_uuid"] = conversation_uuid
        captured["page_num"] = page_request.page_num
        captured["page_size"] = page_request.page_size
        return []

    monkeypatch.setattr(
        assistant_module,
        "conversation_messages_service",
        _fake_conversation_messages,
    )
    client = TestClient(app)

    response = client.get(
        "/admin/assistant/conversation/conv-1/messages",
        headers=_auth_headers(),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 200
    assert body["data"] == []
    assert captured == {"conversation_uuid": "conv-1", "page_num": 1, "page_size": 50}


def test_conversation_messages_route_rejects_page_size_above_50(monkeypatch):
    called = {"value": False}
    _mock_auth(monkeypatch)

    def _fake_conversation_messages(*, conversation_uuid: str, page_request):
        called["value"] = True
        return []

    monkeypatch.setattr(
        assistant_module,
        "conversation_messages_service",
        _fake_conversation_messages,
    )
    client = TestClient(app)

    response = client.get(
        "/admin/assistant/conversation/conv-1/messages",
        headers=_auth_headers(),
        params={"page_size": 51},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == 400
    assert body["message"] == "Validation Failed"
    assert called["value"] is False


def test_conversation_messages_route_forbidden_without_role_or_permission(monkeypatch):
    called = {"value": False}
    _mock_auth(monkeypatch, roles=[], permissions=[])

    def _fake_conversation_messages(*, conversation_uuid: str, page_request):
        called["value"] = True
        return []

    monkeypatch.setattr(
        assistant_module,
        "conversation_messages_service",
        _fake_conversation_messages,
    )
    client = TestClient(app)

    response = client.get(
        "/admin/assistant/conversation/conv-1/messages",
        headers=_auth_headers(),
    )

    assert response.status_code == 403
    body = response.json()
    assert body["code"] == 403
    assert body["message"] == "无权限访问此接口"
    assert called["value"] is False


def test_conversation_messages_route_returns_404_when_conversation_missing(monkeypatch):
    _mock_auth(monkeypatch)

    def _fake_conversation_messages(*, conversation_uuid: str, page_request):
        raise ServiceException(code=ResponseCode.NOT_FOUND, message="会话不存在")

    monkeypatch.setattr(
        assistant_module,
        "conversation_messages_service",
        _fake_conversation_messages,
    )
    client = TestClient(app)

    response = client.get(
        "/admin/assistant/conversation/missing/messages",
        headers=_auth_headers(),
    )

    assert response.status_code == 404
    body = response.json()
    assert body["code"] == 404
    assert body["message"] == "会话不存在"


def test_assistant_route_allows_permission_without_admin_role(monkeypatch):
    captured: dict = {}
    _mock_auth(
        monkeypatch,
        roles=[],
        permissions=["admin:assistant:access"],
    )

    def _fake_assistant_chat(*, question: str, conversation_uuid: str | None = None):
        captured["question"] = question
        captured["conversation_uuid"] = conversation_uuid
        return _build_streaming_response("delegated")

    monkeypatch.setattr(assistant_module, "assistant_chat", _fake_assistant_chat)
    client = TestClient(app)

    response = client.post(
        "/admin/assistant/chat",
        headers=_auth_headers(),
        json={"question": "权限测试"},
    )

    assert response.status_code == 200
    payloads = _extract_payloads(response.text)
    assert payloads[0]["content"]["text"] == "delegated"
    assert captured["question"] == "权限测试"


def test_assistant_route_forbidden_without_role_or_permission(monkeypatch):
    called = {"value": False}
    _mock_auth(monkeypatch, roles=[], permissions=[])

    def _fake_assistant_chat(*, question: str, conversation_uuid: str | None = None):
        called["value"] = True
        return _build_streaming_response("should-not-run")

    monkeypatch.setattr(assistant_module, "assistant_chat", _fake_assistant_chat)
    client = TestClient(app)

    response = client.post(
        "/admin/assistant/chat",
        headers=_auth_headers(),
        json={"question": "forbidden"},
    )

    assert response.status_code == 403
    body = response.json()
    assert body["code"] == 403
    assert body["message"] == "无权限访问此接口"
    assert called["value"] is False


def test_conversation_list_forbidden_without_role_or_permission(monkeypatch):
    called = {"value": False}
    _mock_auth(monkeypatch, roles=[], permissions=[])

    def _fake_chat_list(*, page_request):
        called["value"] = True
        return ([], 0)

    monkeypatch.setattr(
        assistant_module,
        "conversation_list_service",
        _fake_chat_list,
    )
    client = TestClient(app)

    response = client.get(
        "/admin/assistant/conversation/list",
        headers=_auth_headers(),
    )

    assert response.status_code == 403
    body = response.json()
    assert body["code"] == 403
    assert body["message"] == "无权限访问此接口"
    assert called["value"] is False


def test_delete_conversation_route_delegates_to_service(monkeypatch):
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
        "/admin/assistant/conversation/conv-1",
        headers=_auth_headers(),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 200
    assert body["message"] == "删除成功"
    assert body["data"] == {"conversation_uuid": "conv-1"}
    assert captured == {"conversation_uuid": "conv-1"}


def test_update_conversation_title_route_delegates_to_service(monkeypatch):
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
        "/admin/assistant/conversation/conv-1/title",
        headers=_auth_headers(),
        json={"title": "  新标题  "},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 200
    assert body["message"] == "修改成功"
    assert body["data"] == {"conversation_uuid": "conv-1", "title": "新标题"}
    assert captured == {"conversation_uuid": "conv-1", "title": "  新标题  "}


def test_delete_conversation_forbidden_without_role_or_permission(monkeypatch):
    called = {"value": False}
    _mock_auth(monkeypatch, roles=[], permissions=[])

    def _fake_delete_conversation(*, conversation_uuid: str):
        called["value"] = True

    monkeypatch.setattr(
        assistant_module,
        "delete_conversation_service",
        _fake_delete_conversation,
    )
    client = TestClient(app)

    response = client.delete(
        "/admin/assistant/conversation/conv-1",
        headers=_auth_headers(),
    )

    assert response.status_code == 403
    body = response.json()
    assert body["code"] == 403
    assert body["message"] == "无权限访问此接口"
    assert called["value"] is False


def test_update_conversation_title_forbidden_without_role_or_permission(monkeypatch):
    called = {"value": False}
    _mock_auth(monkeypatch, roles=[], permissions=[])

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
        "/admin/assistant/conversation/conv-1/title",
        headers=_auth_headers(),
        json={"title": "新标题"},
    )

    assert response.status_code == 403
    body = response.json()
    assert body["code"] == 403
    assert body["message"] == "无权限访问此接口"
    assert called["value"] is False
