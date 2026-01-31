import pytest

from app.core.codes import ResponseCode
from app.core.exceptions import ServiceException
from app.services import image_parse_service


class DummyResult:
    def __init__(self, content: str) -> None:
        self.content = content


class DummyModel:
    def __init__(self, content: str, capture: dict) -> None:
        self._content = content
        self._capture = capture

    def invoke(self, messages):
        self._capture["messages"] = messages
        return DummyResult(self._content)


def test_parse_drug_images_normalizes_and_parses(monkeypatch):
    capture = {}
    kwargs_capture = {}

    def fake_create_chat_model(**kwargs):
        kwargs_capture.update(kwargs)
        return DummyModel('{"ok": true}', capture)

    monkeypatch.setattr(image_parse_service, "create_chat_model", fake_create_chat_model)

    result = image_parse_service.parse_drug_images(
        ["rawbase64", "data:image/jpeg;base64,abc123"]
    )

    assert result == {"ok": True}
    assert kwargs_capture["model"] == "qwen3-vl-plus"
    assert kwargs_capture["response_format"] == {"type": "json_object"}

    messages = capture["messages"]
    assert len(messages) == 2
    image_parts = messages[1].content
    assert image_parts[0]["image_url"]["url"] == "data:image/png;base64,rawbase64"
    assert image_parts[1]["image_url"]["url"] == "data:image/jpeg;base64,abc123"


def test_parse_drug_images_raises_on_invalid_json(monkeypatch):
    def fake_create_chat_model(**kwargs):
        return DummyModel("not-json", {})

    monkeypatch.setattr(image_parse_service, "create_chat_model", fake_create_chat_model)

    with pytest.raises(ServiceException) as excinfo:
        image_parse_service.parse_drug_images(["raw"])

    assert excinfo.value.code == ResponseCode.INTERNAL_ERROR
    assert excinfo.value.message == "模型返回非 JSON 内容"
