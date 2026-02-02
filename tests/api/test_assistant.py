import json

from fastapi.testclient import TestClient

from app.api.routes import assistant as assistant_module
from app.main import app


class DummyMessage:
    def __init__(self, content: str, tool_calls: list | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


class DummyModel:
    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        return DummyMessage("hello world")


def test_assistant_streaming_response(monkeypatch):
    monkeypatch.setattr(assistant_module, "create_chat_model", lambda: DummyModel())
    client = TestClient(app)

    response = client.post("/api/assistant/chat", json={"question": "hi"})

    assert response.status_code == 200
    lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    assert len(lines) == 2
    payloads = [json.loads(line[len("data: "):]) for line in lines]
    assert payloads[0]["content"] == "hello world"
    assert payloads[0]["is_end"] is False
    assert payloads[1]["content"] == ""
    assert payloads[1]["is_end"] is True


def test_assistant_calls_tool(monkeypatch):
    class DummyTool:
        name = "get_user_info"

        def __init__(self) -> None:
            self.called = False

        async def ainvoke(self, args):
            self.called = True
            return {"name": "Alice"}

    class ToolModel:
        def __init__(self) -> None:
            self.calls = 0

        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return DummyMessage("", tool_calls=[{"name": "get_user_info", "args": {}, "id": "call-1"}])
            return DummyMessage("你好，Alice")

    dummy_tool = DummyTool()
    monkeypatch.setattr(assistant_module, "get_user_info", dummy_tool)
    monkeypatch.setattr(assistant_module, "create_chat_model", lambda: ToolModel())
    client = TestClient(app)

    response = client.post("/api/assistant/chat", json={"question": "当前用户是谁？"})

    assert response.status_code == 200
    assert dummy_tool.called is True
    lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    payloads = [json.loads(line[len("data: "):]) for line in lines]
    assert payloads[0]["content"] == "你好，Alice"


def test_assistant_requires_question():
    client = TestClient(app)

    response = client.post("/api/assistant/chat", json={"question": ""})

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == 400
    assert body["message"] == "问题不能为空"
