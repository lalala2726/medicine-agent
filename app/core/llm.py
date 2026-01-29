import os
from typing import Any, Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

DEFAULT_CHAT_MODEL = "qwen-flash"
DEFAULT_EMBEDDING_MODEL = "text-embedding-v4"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def create_chat_model(
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


def create_embedding_client(
    model: Optional[str] = DEFAULT_EMBEDDING_MODEL,
    api_key: Optional[str] = None,
    base_url: Optional[str] = DEFAULT_BASE_URL,
    dimensions: Optional[int] = 1024,
    **kwargs
) -> OpenAIEmbeddings:
    """
    Create an OpenAI-compatible Embedding client
    (compatible with DashScope OpenAI API).
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
        base_url=base_url,
        dimensions=dimensions,
        **kwargs,
    )
