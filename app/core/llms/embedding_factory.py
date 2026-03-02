from __future__ import annotations

import os
from typing import Any

from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_DASHSCOPE_EMBEDDING_MODEL = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v4")


def create_embedding_model(
        model: str | None = DEFAULT_DASHSCOPE_EMBEDDING_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        dimensions: int | None = 1024,
        **kwargs: Any,
) -> OpenAIEmbeddings:
    """
    功能描述:
        创建 DashScope 兼容的向量嵌入客户端。

    参数说明:
        model (str | None): 嵌入模型名称；默认值 `DASHSCOPE_EMBEDDING_MODEL`。
        api_key (str | None): API 密钥；默认值 `None`。
            为 `None` 时读取环境变量 `DASHSCOPE_API_KEY`。
        base_url (str | None): API 基础地址；默认值 `None`。
            为 `None` 时读取 `DASHSCOPE_BASE_URL`，若未配置则使用 DashScope 默认地址。
        dimensions (int | None): 向量维度；默认值 `1024`。
            当不为 `None` 时需满足范围 [128, 4096] 且为偶数。
        **kwargs (Any): 其余透传 `OpenAIEmbeddings` 构造参数。

    返回值:
        OpenAIEmbeddings: 向量嵌入客户端实例。

    异常说明:
        RuntimeError: 当未提供 `api_key` 且环境变量 `DASHSCOPE_API_KEY` 未设置时抛出。
        ValueError: 当 `dimensions` 不在允许范围或不是偶数时抛出。
    """

    key = api_key or os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set")

    if dimensions is not None:
        if dimensions < 128 or dimensions > 4096 or dimensions % 2 != 0:
            raise ValueError("Dimensions must be between 128 and 4096 and a multiple of 2")

    resolved_base_url = base_url or os.getenv("DASHSCOPE_BASE_URL", DEFAULT_DASHSCOPE_BASE_URL)

    return OpenAIEmbeddings(
        model=model,
        check_embedding_ctx_length=False,
        api_key=SecretStr(key),
        base_url=resolved_base_url,
        dimensions=dimensions,
        **kwargs,
    )
