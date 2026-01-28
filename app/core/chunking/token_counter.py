from __future__ import annotations

import importlib.util
from typing import Optional

from app.core.exceptions import ServiceException


def _get_encoder(encoding_name: Optional[str], model_name: Optional[str]):
    """获取 tiktoken 编码器。"""
    if importlib.util.find_spec("tiktoken") is None:
        raise ServiceException("token 计数依赖 tiktoken，请先安装")
    import tiktoken

    if model_name:
        return tiktoken.encoding_for_model(model_name)
    return tiktoken.get_encoding(encoding_name or "cl100k_base")


def count_tokens(
        text: str,
        *,
        encoding_name: Optional[str] = "cl100k_base",
        model_name: Optional[str] = None,
) -> int:
    """计算文本 token 数量（基于 tiktoken）。"""
    encoder = _get_encoder(encoding_name, model_name)
    return len(encoder.encode(text))
