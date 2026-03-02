from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.core.llms.common import prepare_chat_client_kwargs, resolve_llm_value

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_IMAGE_MODEL = "gpt-4o-mini"


def create_openai_chat_model(
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        extra_body: dict[str, Any] | None = None,
        **kwargs: Any,
) -> ChatOpenAI:
    """
    功能描述:
        创建 OpenAI 兼容聊天模型客户端，并透传调用方提供的扩展参数。

    参数说明:
        model (str | None): 模型名称；默认值 `None`。
            为 `None` 时读取 `OPENAI_CHAT_MODEL`，若未配置则使用 `gpt-4o-mini`。
        api_key (str | None): OpenAI API 密钥；默认值 `None`。
            为 `None` 时读取环境变量 `OPENAI_API_KEY`。
        base_url (str | None): OpenAI API 基础地址；默认值 `None`。
            为 `None` 时读取 `OPENAI_BASE_URL`，若未配置则使用 `https://api.openai.com/v1`。
        extra_body (dict[str, Any] | None): 透传给模型服务端的扩展字段；默认值 `None`。
        **kwargs (Any): 其余透传 `ChatOpenAI` 构造参数。

    返回值:
        ChatOpenAI: 可直接用于 `invoke/stream` 的聊天模型实例。

    异常说明:
        RuntimeError: 当未提供 `api_key` 且环境变量 `OPENAI_API_KEY` 未设置时抛出。
    """

    key = resolve_llm_value(name="OPENAI_API_KEY", explicit=api_key)
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    resolved_model = resolve_llm_value(
        name="OPENAI_CHAT_MODEL",
        explicit=model,
        default=DEFAULT_OPENAI_CHAT_MODEL,
    )
    resolved_base_url = resolve_llm_value(
        name="OPENAI_BASE_URL",
        explicit=base_url,
        default=DEFAULT_OPENAI_BASE_URL,
    )

    prepared_kwargs = prepare_chat_client_kwargs(extra_body=extra_body, **kwargs)

    return ChatOpenAI(
        model=resolved_model,
        api_key=SecretStr(key),
        base_url=resolved_base_url,
        **prepared_kwargs,
    )


def create_openai_image_model(
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        extra_body: dict[str, Any] | None = None,
        **kwargs: Any,
) -> ChatOpenAI:
    """
    功能描述:
        创建 OpenAI 兼容图像理解模型客户端。

    参数说明:
        model (str | None): 图像模型名称；默认值 `None`。
            为 `None` 时读取 `OPENAI_IMAGE_MODEL`，若未配置则使用 `gpt-4o-mini`。
        api_key (str | None): OpenAI API 密钥；默认值 `None`。
            为 `None` 时读取环境变量 `OPENAI_API_KEY`。
        base_url (str | None): OpenAI API 基础地址；默认值 `None`。
            为 `None` 时读取 `OPENAI_BASE_URL`，若未配置则使用 `https://api.openai.com/v1`。
        extra_body (dict[str, Any] | None): 透传扩展字段；默认值 `None`。
        **kwargs (Any): 其余透传 `ChatOpenAI` 构造参数。

    返回值:
        ChatOpenAI: 图像理解可用的聊天模型实例。

    异常说明:
        RuntimeError: 当未提供 `api_key` 且环境变量 `OPENAI_API_KEY` 未设置时抛出。
    """

    resolved_model = resolve_llm_value(
        name="OPENAI_IMAGE_MODEL",
        explicit=model,
        default=DEFAULT_OPENAI_IMAGE_MODEL,
    )
    return create_openai_chat_model(
        model=resolved_model,
        api_key=api_key,
        base_url=base_url,
        extra_body=extra_body,
        **kwargs,
    )
