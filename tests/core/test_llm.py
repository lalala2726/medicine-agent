import pytest
from pydantic import SecretStr

from app.core import llm


class DummyChatOpenAI:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


def test_create_chat_model_requires_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    with pytest.raises(RuntimeError):
        llm.create_chat_model(api_key=None)


def test_create_chat_model_passes_kwargs(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setattr(llm, "ChatOpenAI", DummyChatOpenAI)

    model = llm.create_chat_model(
        model="custom-model",
        base_url="https://example.test",
        response_format={"type": "json_object"},
    )

    assert isinstance(model, DummyChatOpenAI)
    assert model.kwargs["model"] == "custom-model"
    assert isinstance(model.kwargs["api_key"], SecretStr)
    assert model.kwargs["api_key"].get_secret_value() == "test-key"
    assert model.kwargs["base_url"] == "https://example.test"
    assert model.kwargs["model_kwargs"] == {"response_format": {"type": "json_object"}}
