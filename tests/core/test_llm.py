from __future__ import annotations

from typing import Any

import pytest

import app.core.llms.chat_factory as chat_factory
import app.core.llms.embedding_factory as embedding_factory
import app.core.llms.providers.aliyun as aliyun_provider
import app.core.llms.providers.openai as openai_provider
from app.core.llms.provider import LlmProvider


class _FakeChatClient:
    """
    功能描述:
        模拟聊天模型构造对象，用于捕获工厂函数透传参数并断言。

    参数说明:
        **kwargs (Any): 构造时接收的关键字参数。

    返回值:
        None: 仅缓存入参，不执行业务逻辑。

    异常说明:
        无。
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        功能描述:
            初始化模拟对象并保存构造参数。

        参数说明:
            **kwargs (Any): 模型构造参数。

        返回值:
            None

        异常说明:
            无。
        """

        self.kwargs = kwargs


def test_create_chat_model_routes_to_openai_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """测试目的：验证 create_chat_model 在 OPENAI provider 下路由到 OpenAI 工厂；预期结果：返回 OpenAI 工厂产物且参数原样透传。"""

    captured: dict[str, Any] = {}

    def _fake_openai_chat_model(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "openai-model"

    monkeypatch.setattr(chat_factory, "create_openai_chat_model", _fake_openai_chat_model)

    result = chat_factory.create_chat_model(
        model="gpt-test",
        provider=LlmProvider.OPENAI,
        extra_body={"response_format": {"type": "json_object"}},
        temperature=0.2,
    )

    assert result == "openai-model"
    assert captured["model"] == "gpt-test"
    assert captured["extra_body"] == {"response_format": {"type": "json_object"}}
    assert captured["temperature"] == 0.2


def test_create_chat_model_routes_to_aliyun_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """测试目的：验证 create_chat_model 在 ALIYUN provider 下路由到阿里云工厂；预期结果：返回阿里云工厂产物且参数原样透传。"""

    captured: dict[str, Any] = {}

    def _fake_aliyun_chat_model(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "aliyun-model"

    monkeypatch.setattr(chat_factory, "create_aliyun_chat_model", _fake_aliyun_chat_model)

    result = chat_factory.create_chat_model(
        model="qwen-test",
        provider=LlmProvider.ALIYUN,
        extra_body={"enable_thinking": True},
        think=True,
    )

    assert result == "aliyun-model"
    assert captured["model"] == "qwen-test"
    assert captured["extra_body"] == {"enable_thinking": True}
    assert captured["think"] is True


def test_create_chat_model_accepts_provider_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """测试目的：验证 create_chat_model 支持 provider 字符串输入；预期结果：字符串 `aliyun` 可被识别并正确路由。"""

    monkeypatch.setattr(chat_factory, "create_aliyun_chat_model", lambda **_: "aliyun-model")

    result = chat_factory.create_chat_model(provider="aliyun")

    assert result == "aliyun-model"


def test_create_chat_model_raises_for_invalid_provider() -> None:
    """测试目的：验证 create_chat_model 对非法 provider 输入报错；预期结果：抛出 ValueError。"""

    with pytest.raises(ValueError):
        chat_factory.create_chat_model(provider="unknown")


def test_create_openai_chat_model_reads_env_and_enables_stream_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    """测试目的：验证 OpenAI 聊天工厂读取 OPENAI 环境变量并默认启用 stream_usage；预期结果：构造参数包含环境值且 `stream_usage=True`。"""

    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-env")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai.example.com/v1")
    monkeypatch.setattr(openai_provider, "ChatOpenAI", _FakeChatClient)

    model = openai_provider.create_openai_chat_model(extra_body={"enable_thinking": True})

    assert isinstance(model, _FakeChatClient)
    assert model.kwargs["model"] == "gpt-env"
    assert model.kwargs["base_url"] == "https://openai.example.com/v1"
    assert model.kwargs["extra_body"] == {"enable_thinking": True}
    assert model.kwargs["stream_usage"] is True


def test_create_openai_chat_model_raises_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """测试目的：验证 OpenAI 聊天工厂在缺少 API Key 时失败；预期结果：抛出 RuntimeError。"""

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        openai_provider.create_openai_chat_model(model="gpt-test")


def test_create_aliyun_chat_model_reads_env_and_enables_stream_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    """测试目的：验证阿里云聊天工厂读取 DASHSCOPE 环境变量并默认启用 stream_usage；预期结果：构造参数包含环境值且 `stream_usage=True`。"""

    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-key")
    monkeypatch.setenv("DASHSCOPE_CHAT_MODEL", "qwen-env")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://dashscope.example.com/v1")
    monkeypatch.setattr(aliyun_provider, "ChatQwen", _FakeChatClient)

    model = aliyun_provider.create_aliyun_chat_model(extra_body={"enable_thinking": True})

    assert isinstance(model, _FakeChatClient)
    assert model.kwargs["model"] == "qwen-env"
    assert model.kwargs["base_url"] == "https://dashscope.example.com/v1"
    assert model.kwargs["extra_body"] == {"enable_thinking": True}
    assert model.kwargs["stream_usage"] is True


def test_create_aliyun_chat_model_raises_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """测试目的：验证阿里云聊天工厂在缺少 API Key 时失败；预期结果：抛出 RuntimeError。"""

    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY is not set"):
        aliyun_provider.create_aliyun_chat_model(model="qwen-test")


def test_create_image_model_routes_to_aliyun_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """测试目的：验证 create_image_model 在 ALIYUN provider 下正确路由；预期结果：返回阿里云图像工厂产物且参数原样透传。"""

    captured: dict[str, Any] = {}

    def _fake_aliyun_image_model(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "aliyun-image-model"

    monkeypatch.setattr(chat_factory, "create_aliyun_image_model", _fake_aliyun_image_model)

    result = chat_factory.create_image_model(
        model="qwen3-vl-plus",
        provider=LlmProvider.ALIYUN,
        extra_body={"response_format": {"type": "json_object"}},
        think=True,
    )

    assert result == "aliyun-image-model"
    assert captured["model"] == "qwen3-vl-plus"
    assert captured["extra_body"] == {"response_format": {"type": "json_object"}}
    assert captured["think"] is True


def test_create_embedding_model_validates_dimensions_and_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """测试目的：验证嵌入模型工厂参数校验逻辑；预期结果：缺少 API Key 抛 RuntimeError，非法 dimensions 抛 ValueError。"""

    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY is not set"):
        embedding_factory.create_embedding_model()

    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-key")
    with pytest.raises(ValueError, match="Dimensions must be between 128 and 4096 and a multiple of 2"):
        embedding_factory.create_embedding_model(dimensions=127)
