import os
from typing import Any, Optional

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

DEFAULT_MODEL = "qwen-flash"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def get_chat_model(
    model: str = DEFAULT_MODEL,
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
    return ChatOpenAI(
        model=model,
        api_key=SecretStr(key),
        base_url=base_url or DEFAULT_BASE_URL,
        model_kwargs=model_kwargs or None,
    )
