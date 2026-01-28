import os
from typing import Any, Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

DEFAULT_CHAT_MODEL = "qwen-flash"
DEFAULT_EMBEDDING_MODEL = "text-embedding-v4"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def get_chat_model(
        model: str = DEFAULT_CHAT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        response_format: Optional[dict[str, Any]] = None,
) -> ChatOpenAI:
    key = api_key or os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set")
    model_kwargs = {}
    if response_format:
        model_kwargs["response_format"] = response_format
    kwargs = {}
    if model_kwargs:
        kwargs["model_kwargs"] = model_kwargs
    return ChatOpenAI(
        model=model,
        api_key=SecretStr(key),
        base_url=base_url or DEFAULT_BASE_URL,
        **kwargs,
    )


def get_embedding_model(
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dimensions: Optional[int] = None,
        organization: Optional[str] = None,
        max_retries: Optional[int] = None,
        request_timeout: Optional[float] = None,
) -> OpenAIEmbeddings:
    """
    构建 OpenAIEmbeddings 客户端（兼容 DashScope 的 OpenAI API）。

    Args:
        model: 模型名称（默认读取环境变量）
        api_key: API Key（默认读取环境变量）
        base_url: API Base URL（默认读取环境变量）
        dimensions: 期望的向量维度（text-embedding-3 以上支持）
        organization: OpenAI 组织 ID（可选）
        max_retries: 最大重试次数
        request_timeout: 请求超时（秒）
    """
    if dimensions is not None:
        min_dimensions = 128
        max_dimensions = 4096
        if dimensions < min_dimensions or dimensions > max_dimensions or dimensions % 2 != 0:
            raise ValueError(
                f"Invalid dimensions: {dimensions}. It should be an even integer between {min_dimensions} and {max_dimensions}."
            )

    key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY/DASHSCOPE_API_KEY is not set")

    resolved_model = (
            model
            or os.getenv("OPENAI_EMBEDDING_MODEL")
            or os.getenv("DASHSCOPE_EMBEDDING_MODEL")
            or DEFAULT_EMBEDDING_MODEL
    )
    resolved_base_url = (
            base_url
            or os.getenv("OPENAI_API_BASE")
            or os.getenv("DASHSCOPE_BASE_URL")
            or DEFAULT_BASE_URL
    )
    resolved_org = organization or os.getenv("OPENAI_ORG_ID") or os.getenv("OPENAI_ORGANIZATION")

    kwargs: dict[str, Any] = {
        "model": resolved_model,
        "api_key": SecretStr(key),
        "base_url": resolved_base_url,
    }
    if dimensions is not None:
        kwargs["dimensions"] = dimensions
    if resolved_org:
        kwargs["organization"] = resolved_org
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    if request_timeout is not None:
        kwargs["request_timeout"] = request_timeout

    return OpenAIEmbeddings(**kwargs)
