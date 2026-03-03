from __future__ import annotations

from typing import Dict

from app.core.exception.exceptions import ServiceException
from app.rag.chunking.strategies import (
    CharacterChunker,
    MarkdownHeaderChunker,
    RecursiveChunker,
    TokenChunker,
)
from app.rag.chunking.types import ChunkStrategy, ChunkStrategyType


class ChunkerFactory:
    """
    功能描述:
        切片策略工厂，负责切片类型与策略实例之间的注册和获取。

    参数说明:
        无。通过类方法操作策略注册表。

    返回值:
        无。通过 get 方法返回策略实例。

    异常说明:
        ServiceException: 获取未注册策略时抛出。
    """

    _registry: Dict[str, ChunkStrategy] = {}

    @classmethod
    def register(cls, name: str | ChunkStrategyType, strategy: ChunkStrategy) -> None:
        """
        功能描述:
            注册切片策略，支持后续扩展新的策略实现。

        参数说明:
            name (str | ChunkStrategyType): 策略名称或策略枚举值。
            strategy (ChunkStrategy): 策略实例对象。

        返回值:
            None: 注册完成无返回值。

        异常说明:
            无。
        """
        key = name.value if isinstance(name, ChunkStrategyType) else name
        cls._registry[key] = strategy

    @classmethod
    def get(cls, name: str | ChunkStrategyType) -> ChunkStrategy:
        """
        功能描述:
            按策略名称或枚举值获取对应策略实例。

        参数说明:
            name (str | ChunkStrategyType): 策略名称或策略枚举值。

        返回值:
            ChunkStrategy: 对应切片策略实例。

        异常说明:
            ServiceException: 当策略未注册或名称非法时抛出。
        """
        key = name.value if isinstance(name, ChunkStrategyType) else name
        strategy = cls._registry.get(key)
        if not strategy:
            raise ServiceException(f"不支持的切片方式: {name}")
        return strategy


# 默认策略注册
ChunkerFactory.register(ChunkStrategyType.CHARACTER, CharacterChunker())
ChunkerFactory.register(ChunkStrategyType.RECURSIVE, RecursiveChunker())
ChunkerFactory.register(ChunkStrategyType.TOKEN, TokenChunker())
ChunkerFactory.register(ChunkStrategyType.MARKDOWN_HEADER, MarkdownHeaderChunker())
