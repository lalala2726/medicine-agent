from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain.agents.structured_output import ToolStrategy

from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException
from app.services import image_parse_service


def test_parse_drug_images_normalizes_and_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    """测试目的：验证图片解析走 create_agent + ToolStrategy 链路；预期结果：结构化结果成功返回且图片输入被规范化。"""

    create_image_model_kwargs: dict = {}
    create_agent_kwargs: dict = {}
    captured_messages: dict = {}

    fake_llm = object()
    fake_agent = object()

    def fake_create_image_model(**kwargs):
        create_image_model_kwargs.update(kwargs)
        return fake_llm

    def fake_create_agent(**kwargs):
        create_agent_kwargs.update(kwargs)
        return fake_agent

    def fake_agent_invoke(agent_instance, history_messages):
        captured_messages["agent_instance"] = agent_instance
        captured_messages["history_messages"] = history_messages
        return SimpleNamespace(
            payload={"structured_response": {"commonName": "阿司匹林"}},
            content="",
        )

    monkeypatch.setattr(image_parse_service, "create_image_model", fake_create_image_model)
    monkeypatch.setattr(image_parse_service, "create_agent", fake_create_agent)
    monkeypatch.setattr(image_parse_service, "agent_invoke", fake_agent_invoke)

    result = image_parse_service.parse_drug_images(
        ["rawbase64", "data:image/jpeg;base64,abc123"],
    )

    assert result["commonName"] == "阿司匹林"
    assert "warmTips" in result
    assert create_image_model_kwargs["model"] == "qwen3-vl-plus"
    assert create_agent_kwargs["model"] is fake_llm
    assert create_agent_kwargs["system_prompt"].content == image_parse_service._DRUG_PARSER_PROMPT
    response_format = create_agent_kwargs["response_format"]
    assert isinstance(response_format, ToolStrategy)
    assert response_format.schema is image_parse_service.DrugImageSchema
    assert captured_messages["agent_instance"] is fake_agent

    history_messages = captured_messages["history_messages"]
    assert len(history_messages) == 1
    image_parts = history_messages[0].content
    assert image_parts[0]["image_url"]["url"] == "data:image/png;base64,rawbase64"
    assert image_parts[1]["image_url"]["url"] == "data:image/jpeg;base64,abc123"


def test_parse_drug_images_raises_on_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """测试目的：验证结构化结果缺失时抛出业务异常；预期结果：抛出 INTERNAL_ERROR 且提示模型返回非 JSON。"""

    monkeypatch.setattr(image_parse_service, "create_image_model", lambda **_kwargs: object())
    monkeypatch.setattr(image_parse_service, "create_agent", lambda **_kwargs: object())
    monkeypatch.setattr(
        image_parse_service,
        "agent_invoke",
        lambda _agent, _messages: SimpleNamespace(payload={}, content=""),
    )

    with pytest.raises(ServiceException) as excinfo:
        image_parse_service.parse_drug_images(["raw"])

    assert excinfo.value.code == ResponseCode.INTERNAL_ERROR
    assert excinfo.value.message == "模型返回非 JSON 内容"

