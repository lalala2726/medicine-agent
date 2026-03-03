from __future__ import annotations

from typing import Optional

from app.rag.chunking.factory import ChunkerFactory
from app.rag.chunking.types import ChunkStrategyType, SplitChunk, SplitConfig


def split_text(
    text: str,
    strategy_type: str | ChunkStrategyType,
    config: Optional[SplitConfig] = None,
) -> list[SplitChunk]:
    """
    功能描述:
        对输入文本应用指定切片策略并返回结构化切片列表。

    参数说明:
        text (str): 待切片文本。
        strategy_type (str | ChunkStrategyType): 切片策略类型。
        config (SplitConfig | None): 切片配置，默认值为 None；为空时使用默认配置。

    返回值:
        list[SplitChunk]: 切片结果列表。

    异常说明:
        ServiceException: strategy_type 未注册时由工厂抛出。
        ServiceException: 具体策略执行失败时由策略实现抛出。
    """
    if not text:
        return []
    resolved_config = config or SplitConfig()
    strategy = ChunkerFactory.get(strategy_type)
    return strategy.split_text(text, resolved_config)
