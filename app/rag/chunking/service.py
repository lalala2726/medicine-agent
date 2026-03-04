from __future__ import annotations

import logging
from typing import Optional

from app.rag.chunking.factory import ChunkerFactory
from app.rag.chunking.types import ChunkStrategyType, SplitChunk, SplitConfig

_CHUNK_OVERSIZE_WARNING_PREFIX = "Created a chunk of size"


class _LangChainChunkWarningFilter(logging.Filter):
    """
    功能描述:
        过滤 LangChain Text Splitter 的超长 chunk 提示日志，避免控制台刷屏。

    参数说明:
        name (str): logging.Filter 基类参数，默认值为 ""。

    返回值:
        无。该类用于日志过滤。

    异常说明:
        无。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """
        功能描述:
            判断日志记录是否允许输出。

        参数说明:
            record (logging.LogRecord): 待过滤日志记录。

        返回值:
            bool: 允许输出返回 True；命中超长 chunk 提示返回 False。

        异常说明:
            无。
        """
        return not record.getMessage().startswith(_CHUNK_OVERSIZE_WARNING_PREFIX)


def _suppress_langchain_chunk_size_warning() -> None:
    """
    功能描述:
        对 `langchain_text_splitters.base` 注入过滤器，仅屏蔽超长 chunk 告警日志。

    参数说明:
        无。

    返回值:
        None: 初始化过滤器后无返回值。

    异常说明:
        无。
    """
    logger = logging.getLogger("langchain_text_splitters.base")
    if any(isinstance(item, _LangChainChunkWarningFilter) for item in logger.filters):
        return
    logger.addFilter(_LangChainChunkWarningFilter())


_suppress_langchain_chunk_size_warning()


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
