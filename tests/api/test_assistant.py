import json

from fastapi.testclient import TestClient

from app.api.routes import admin_assistant as assistant_module
from app.main import app


class _DummyGraph:
    def __init__(self, final_state: dict | None = None, should_raise: bool = False) -> None:
        self.final_state = final_state or {}
        self.should_raise = should_raise

    def invoke(self, _state: dict) -> dict:
        if self.should_raise:
            raise RuntimeError("graph failed")
        return self.final_state


def test_assistant_streaming_response_from_workflow_chat_result(monkeypatch):
    monkeypatch.setattr(
        assistant_module,
        "ADMIN_WORKFLOW",
        _DummyGraph(final_state={"results": {"chat": {"content": "hello from workflow"}}}),
    )
    client = TestClient(app)

    response = client.post("/api/admin/assistant/chat", json={"question": "hi"})

    assert response.status_code == 200
    lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    assert len(lines) == 2
    payloads = [json.loads(line[len("data: "):]) for line in lines]
    assert payloads[0]["content"] == "hello from workflow"
    assert payloads[0]["is_end"] is False
    assert payloads[1]["content"] == ""
    assert payloads[1]["is_end"] is True


def test_assistant_streaming_response_from_order_context(monkeypatch):
    monkeypatch.setattr(
        assistant_module,
        "ADMIN_WORKFLOW",
        _DummyGraph(final_state={"order_context": {"result": {"content": "order done"}}}),
    )
    client = TestClient(app)

    response = client.post("/api/admin/assistant/chat", json={"question": "查订单"})

    assert response.status_code == 200
    lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    payloads = [json.loads(line[len("data: "):]) for line in lines]
    assert payloads[0]["content"] == "order done"
    assert payloads[1]["is_end"] is True


def test_assistant_streaming_response_when_workflow_raises(monkeypatch):
    monkeypatch.setattr(
        assistant_module,
        "ADMIN_WORKFLOW",
        _DummyGraph(should_raise=True),
    )
    client = TestClient(app)

    response = client.post("/api/admin/assistant/chat", json={"question": "hi"})

    assert response.status_code == 200
    lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    payloads = [json.loads(line[len("data: "):]) for line in lines]
    assert payloads[0]["content"] == "服务暂时不可用，请稍后重试。"
    assert payloads[1]["is_end"] is True


def test_assistant_requires_question():
    client = TestClient(app)

    response = client.post("/api/admin/assistant/chat", json={"question": ""})

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == 400
    assert body["message"] == "问题不能为空"
