from __future__ import annotations

from typing import Any, TypeAlias

from langchain_openai import ChatOpenAI

from app.core.llms.provider import LlmProvider, normalize_provider
from app.core.llms.providers import (
    ChatQwen,
    create_aliyun_chat_model,
    create_aliyun_image_model,
    create_openai_chat_model,
    create_openai_image_model,
)

ChatModel: TypeAlias = ChatOpenAI | ChatQwen


def create_chat_model(
        *,
        model: str | None = None,
        provider: LlmProvider = LlmProvider.OPENAI,
        api_key: str | None = None,
        base_url: str | None = None,
        extra_body: dict[str, Any] | None = None,
        **kwargs: Any,
) -> ChatModel:
    """
    功能描述:
        按厂商创建聊天模型客户端。

    参数说明:
        model (str | None): 模型名称；默认值 `None`。
            为 `None` 时使用对应 provider 的默认模型。
        provider (LlmProvider): 模型厂商；默认值 `LlmProvider.OPENAI`。
            支持 `openai` 与 `aliyun`。
        api_key (str | None): 覆盖厂商 API 密钥；默认值 `None`。
        base_url (str | None): 覆盖厂商 API Base URL；默认值 `None`。
        extra_body (dict[str, Any] | None): 扩展请求体字段；默认值 `None`。
        **kwargs (Any): 其余透传底层模型构造参数。

    返回值:
        ChatModel: 根据 provider 返回 `ChatOpenAI` 或 `ChatQwen`。

    异常说明:
        ValueError: 当 provider 取值不受支持时抛出。
        RuntimeError: 当对应 provider 的必填密钥缺失时由底层抛出。
    """

    resolved_provider = normalize_provider(provider)
    if resolved_provider is LlmProvider.ALIYUN:
        return create_aliyun_chat_model(
            model=model,
            api_key=api_key,
            base_url=base_url,
            extra_body=extra_body,
            **kwargs,
        )
    if resolved_provider is LlmProvider.OPENAI:
        return create_openai_chat_model(
            model=model,
            api_key=api_key,
            base_url=base_url,
            extra_body=extra_body,
            **kwargs,
        )
    raise ValueError(f"Unsupported provider: {resolved_provider}")


def create_image_model(
        *,
        model: str | None = None,
        provider: LlmProvider = LlmProvider.OPENAI,
        api_key: str | None = None,
        base_url: str | None = None,
        extra_body: dict[str, Any] | None = None,
        **kwargs: Any,
) -> ChatModel:
    """
    功能描述:
        按厂商创建图像理解模型客户端。

    参数说明:
        model (str | None): 模型名称；默认值 `None`。
            为 `None` 时使用对应 provider 的默认图像模型。
        provider (LlmProvider): 模型厂商；默认值 `LlmProvider.OPENAI`。
            支持 `openai` 与 `aliyun`。
        api_key (str | None): 覆盖厂商 API 密钥；默认值 `None`。
        base_url (str | None): 覆盖厂商 API Base URL；默认值 `None`。
        extra_body (dict[str, Any] | None): 扩展请求体字段；默认值 `None`。
        **kwargs (Any): 其余透传底层模型构造参数。

    返回值:
        ChatModel: 根据 provider 返回 `ChatOpenAI` 或 `ChatQwen`。

    异常说明:
        ValueError: 当 provider 取值不受支持时抛出。
        RuntimeError: 当对应 provider 的必填密钥缺失时由底层抛出。
    """

    resolved_provider = normalize_provider(provider)
    if resolved_provider is LlmProvider.ALIYUN:
        return create_aliyun_image_model(
            model=model,
            api_key=api_key,
            base_url=base_url,
            extra_body=extra_body,
            **kwargs,
        )
    if resolved_provider is LlmProvider.OPENAI:
        return create_openai_image_model(
            model=model,
            api_key=api_key,
            base_url=base_url,
            extra_body=extra_body,
            **kwargs,
        )
    raise ValueError(f"Unsupported provider: {resolved_provider}")
