from __future__ import annotations

from typing import Any, TypeAlias

from langchain_openai import ChatOpenAI

from app.core.llms.provider import LlmProvider, resolve_provider
from app.core.llms.providers import (
    ChatQwen,
    ChatVolcengine,
    create_aliyun_chat_model,
    create_aliyun_image_model,
    create_openai_chat_model,
    create_openai_image_model,
    create_volcengine_chat_model,
    create_volcengine_image_model,
)
from app.core.llms.common import resolve_provider_extra_body

ChatModel: TypeAlias = ChatOpenAI | ChatQwen | ChatVolcengine


def create_chat_model(
        *,
        model: str | None = None,
        provider: LlmProvider | str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        extra_body: dict[str, Any] | None = None,
        think: bool = False,
        **kwargs: Any,
) -> ChatModel:
    """
    功能描述:
        按厂商创建聊天模型客户端。

    参数说明:
        model (str | None): 模型名称；默认值 `None`。
            为 `None` 时使用对应 provider 的默认模型。
        provider (LlmProvider | str | None): 模型厂商；默认值 `None`。
            优先使用显式入参；未传时读取 `LLM_PROVIDER`，再回退到 `openai`。
        api_key (str | None): 覆盖厂商 API 密钥；默认值 `None`。
        base_url (str | None): 覆盖厂商 API Base URL；默认值 `None`。
        extra_body (dict[str, Any] | None): 扩展请求体字段；默认值 `None`。
        think (bool): 是否开启深度思考；默认值 `False`。
            开关将按 provider 语义映射到请求体对应字段。
        **kwargs (Any): 其余透传底层模型构造参数。

    返回值:
        ChatModel: 根据 provider 返回 `ChatOpenAI`、`ChatQwen` 或 `ChatVolcengine`。

    异常说明:
        ValueError: 当 provider 取值不受支持时抛出。
        RuntimeError: 当对应 provider 的必填密钥缺失时由底层抛出。
    """

    resolved_provider = resolve_provider(provider)
    resolved_extra_body = resolve_provider_extra_body(
        provider=resolved_provider,
        extra_body=extra_body,
        think=think,
    )
    if resolved_provider is LlmProvider.ALIYUN:
        return create_aliyun_chat_model(
            model=model,
            api_key=api_key,
            base_url=base_url,
            extra_body=resolved_extra_body,
            **kwargs,
        )
    if resolved_provider is LlmProvider.VOLCENGINE:
        return create_volcengine_chat_model(
            model=model,
            api_key=api_key,
            base_url=base_url,
            extra_body=resolved_extra_body,
            **kwargs,
        )
    if resolved_provider is LlmProvider.OPENAI:
        return create_openai_chat_model(
            model=model,
            api_key=api_key,
            base_url=base_url,
            extra_body=resolved_extra_body,
            **kwargs,
        )
    raise ValueError(f"Unsupported provider: {resolved_provider}")


def create_image_model(
        *,
        model: str | None = None,
        provider: LlmProvider | str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        extra_body: dict[str, Any] | None = None,
        think: bool = False,
        **kwargs: Any,
) -> ChatModel:
    """
    功能描述:
        按厂商创建图像理解模型客户端。

    参数说明:
        model (str | None): 模型名称；默认值 `None`。
            为 `None` 时使用对应 provider 的默认图像模型。
        provider (LlmProvider | str | None): 模型厂商；默认值 `None`。
            优先使用显式入参；未传时读取 `LLM_PROVIDER`，再回退到 `openai`。
        api_key (str | None): 覆盖厂商 API 密钥；默认值 `None`。
        base_url (str | None): 覆盖厂商 API Base URL；默认值 `None`。
        extra_body (dict[str, Any] | None): 扩展请求体字段；默认值 `None`。
        think (bool): 是否开启深度思考；默认值 `False`。
            开关将按 provider 语义映射到请求体对应字段。
        **kwargs (Any): 其余透传底层模型构造参数。

    返回值:
        ChatModel: 根据 provider 返回 `ChatOpenAI`、`ChatQwen` 或 `ChatVolcengine`。

    异常说明:
        ValueError: 当 provider 取值不受支持时抛出。
        RuntimeError: 当对应 provider 的必填密钥缺失时由底层抛出。
    """

    resolved_provider = resolve_provider(provider)
    resolved_extra_body = resolve_provider_extra_body(
        provider=resolved_provider,
        extra_body=extra_body,
        think=think,
    )
    if resolved_provider is LlmProvider.ALIYUN:
        return create_aliyun_image_model(
            model=model,
            api_key=api_key,
            base_url=base_url,
            extra_body=resolved_extra_body,
            **kwargs,
        )
    if resolved_provider is LlmProvider.VOLCENGINE:
        return create_volcengine_image_model(
            model=model,
            api_key=api_key,
            base_url=base_url,
            extra_body=resolved_extra_body,
            **kwargs,
        )
    if resolved_provider is LlmProvider.OPENAI:
        return create_openai_image_model(
            model=model,
            api_key=api_key,
            base_url=base_url,
            extra_body=resolved_extra_body,
            **kwargs,
        )
    raise ValueError(f"Unsupported provider: {resolved_provider}")
