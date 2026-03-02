from __future__ import annotations

import os
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

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

    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    resolved_model = model or os.getenv("OPENAI_CHAT_MODEL", DEFAULT_OPENAI_CHAT_MODEL)
    resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)

    raw_model_kwargs = kwargs.pop("model_kwargs", None)
    model_kwargs = dict(raw_model_kwargs or {})
    if model_kwargs:
        kwargs["model_kwargs"] = model_kwargs

    resolved_extra_body = dict(extra_body or {})
    if resolved_extra_body:
        kwargs["extra_body"] = resolved_extra_body

    kwargs.setdefault("stream_usage", True)

    return ChatOpenAI(
        model=resolved_model,
        api_key=SecretStr(key),
        base_url=resolved_base_url,
        **kwargs,
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

    resolved_model = model or os.getenv("OPENAI_IMAGE_MODEL", DEFAULT_OPENAI_IMAGE_MODEL)
    return create_openai_chat_model(
        model=resolved_model,
        api_key=api_key,
        base_url=base_url,
        extra_body=extra_body,
        **kwargs,
    )
