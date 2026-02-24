from __future__ import annotations

from typing import Dict

from app.rag.chunking.base import ChunkStrategy, ChunkStrategyType
from app.rag.chunking.length_splitter import LengthChunker
from app.rag.chunking.recursive_splitter import RecursiveChunker
from app.rag.chunking.title_splitter import TitleChunker
from app.rag.chunking.token_splitter import TokenChunker
from app.core.exception.exceptions import ServiceException


class ChunkerFactory:
    """切片策略工厂，统一管理注册与获取。"""

    _registry: Dict[str, ChunkStrategy] = {}

    @classmethod
    def register(cls, name: str, strategy: ChunkStrategy) -> None:
        """
        注册切片策略，便于扩展新的切片方式。

        Args:
            name: 策略名称
            strategy: 策略实例
        """
        cls._registry[name] = strategy

    @classmethod
    def get(cls, name: str | ChunkStrategyType) -> ChunkStrategy:
        """
        获取切片策略。

        Args:
            name: 策略名称

        Returns:
            对应策略实例
        """
        key = name.value if isinstance(name, ChunkStrategyType) else name
        strategy = cls._registry.get(key)
        if not strategy:
            raise ServiceException(f"不支持的切片方式: {name}")
        return strategy


# 默认策略注册
ChunkerFactory.register("length", LengthChunker())
ChunkerFactory.register("title", TitleChunker())
ChunkerFactory.register("token", TokenChunker())
ChunkerFactory.register("recursive", RecursiveChunker())
