import json

from fastapi.testclient import TestClient

from app.main import app
from app.api.routes import assistant as assistant_module


class DummyChunk:
    def __init__(self, content: str) -> None:
        self.content = content


class DummyModel:
    async def astream(self, prompt: str):
        for content in ["hello", "world"]:
            yield DummyChunk(content)


def test_assistant_streaming_response(monkeypatch):
    monkeypatch.setattr(assistant_module, "get_chat_model", lambda: DummyModel())
    client = TestClient(app)

    response = client.post("/api/assistant/chat", json={"question": "hi"})

    assert response.status_code == 200
    lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    assert len(lines) == 3
    payloads = [json.loads(line[len("data: "):]) for line in lines]
    assert payloads[0]["content"] == "hello"
    assert payloads[0]["is_end"] is False
    assert payloads[1]["content"] == "world"
    assert payloads[1]["is_end"] is False
    assert payloads[2]["content"] == ""
    assert payloads[2]["is_end"] is True


def test_assistant_requires_question():
    client = TestClient(app)

    response = client.post("/api/assistant/chat", json={"question": ""})

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == 400
    assert body["message"] == "问题不能为空"
