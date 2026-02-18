import os
from typing import Any, Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

# Load from environment variables
DEFAULT_CHAT_MODEL = os.getenv("DASHSCOPE_CHAT_MODEL", "qwen-max")
DEFAULT_IMAGE_MODEL = os.getenv("DASHSCOPE_IMAGE_MODEL", "qwen3-vl-flash")
DEFAULT_EMBEDDING_MODEL = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v4")
DEFAULT_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")


def create_chat_model(
        model: str = DEFAULT_CHAT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        response_format: Optional[dict[str, Any]] = None,
        **kwargs
) -> ChatOpenAI:
    """
    创建聊天模型客户端

    Args:
        model: 模型名称，默认使用环境变量 DASHSCOPE_CHAT_MODEL
        api_key: API密钥，默认使用环境变量 DASHSCOPE_API_KEY
        base_url: API基础URL，默认使用环境变量 DASHSCOPE_BASE_URL
        response_format: 响应格式配置，如 {"type": "json_object"}
        **kwargs: 其他传递给 ChatOpenAI 的参数

    Returns:
        ChatOpenAI: 聊天模型客户端

    Raises:
        RuntimeError: 当 API_KEY 未设置时
    """
    key = api_key or os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set")

    model_kwargs = {}
    if response_format:
        model_kwargs["response_format"] = response_format
    if model_kwargs:
        kwargs["model_kwargs"] = model_kwargs
    # 默认开启 stream_usage，优先获取模型返回的真实 token 消耗。
    # 若供应商未返回 usage，业务层会使用 tiktoken 估算兜底。
    kwargs.setdefault("stream_usage", True)

    return ChatOpenAI(
        model=model,
        api_key=SecretStr(key),
        base_url=base_url or DEFAULT_BASE_URL,
        **kwargs,
    )


def create_embedding_model(
        model: Optional[str] = DEFAULT_EMBEDDING_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dimensions: Optional[int] = 1024,
        **kwargs
) -> OpenAIEmbeddings:
    """
    创建向量嵌入客户端

    Args:
        model: 嵌入模型名称，默认使用环境变量 DASHSCOPE_EMBEDDING_MODEL
        api_key: API密钥，默认使用环境变量 DASHSCOPE_API_KEY
        base_url: API基础URL，默认使用环境变量 DASHSCOPE_BASE_URL
        dimensions: 向量维度，范围 128-4096 且为偶数
        **kwargs: 其他传递给 OpenAIEmbeddings 的参数

    Returns:
        OpenAIEmbeddings: 向量嵌入客户端

    Raises:
        RuntimeError: 当 API_KEY 未设置时
        ValueError: 当维度不在有效范围时
    """
    key = api_key or os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set")

    if dimensions is not None:
        if dimensions < 128 or dimensions > 4096 or dimensions % 2 != 0:
            raise ValueError("Dimensions must be between 128 and 4096 and a multiple of 2")

    return OpenAIEmbeddings(
        model=model,
        check_embedding_ctx_length=False,
        api_key=SecretStr(key),
        base_url=base_url or DEFAULT_BASE_URL,
        dimensions=dimensions,
        **kwargs,
    )


def create_image_model(
        model: str = DEFAULT_IMAGE_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        response_format: Optional[dict[str, Any]] = None,
        **kwargs
) -> ChatOpenAI:
    """
    创建图像理解模型客户端

    Args:
        model: 模型名称，默认使用环境变量 DASHSCOPE_IMAGE_MODEL
        api_key: API密钥
        base_url: API基础URL
        response_format: 响应格式配置
        **kwargs: 其他参数

    Returns:
        ChatOpenAI: 图像模型客户端
    """
    return create_chat_model(
        model=model,
        api_key=api_key,
        base_url=base_url,
        response_format=response_format,
        **kwargs
    )
