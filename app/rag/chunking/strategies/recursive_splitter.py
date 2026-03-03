from __future__ import annotations

import importlib.util

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.exception.exceptions import ServiceException
from app.rag.chunking.types import (
    ChunkStrategy,
    SplitChunk,
    SplitConfig,
    build_chunk_stats,
)


class RecursiveChunker(ChunkStrategy):
    """
    功能描述:
        基于递归分隔符的文本切片策略，优先保留语义边界（段落、换行、空格）。

    参数说明:
        无。配置由 split_text 方法参数传入。

    返回值:
        无。通过 split_text 返回切片结果。

    异常说明:
        ServiceException: 当启用 tiktoken 但环境未安装相关依赖时抛出。
    """

    def split_text(self, text: str, config: SplitConfig) -> list[SplitChunk]:
        """
        功能描述:
            按递归分隔符策略切分单段文本，生成标准切片结构。

        参数说明:
            text (str): 待切片文本。
            config (SplitConfig): 切片配置。

        返回值:
            list[SplitChunk]: 切片结果列表，每项包含文本和字符统计信息。

        异常说明:
            ServiceException: 当 use_tiktoken=True 且未安装 tiktoken 时抛出。
        """
        splitter = self._build_splitter(config)
        pieces = splitter.split_text(text)
        return [
            SplitChunk(
                text=piece,
                stats=build_chunk_stats(piece),
            )
            for piece in pieces
        ]

    def _build_splitter(self, config: SplitConfig) -> RecursiveCharacterTextSplitter:
        """
        功能描述:
            根据配置构建递归字符切片器。

        参数说明:
            config (SplitConfig): 切片配置。

        返回值:
            RecursiveCharacterTextSplitter: 递归字符切片器实例。

        异常说明:
            ServiceException: 当 use_tiktoken=True 且未安装 tiktoken 时抛出。
        """
        if config.use_tiktoken:
            self._ensure_tiktoken()
            if config.model_name:
                return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                    model_name=config.model_name,
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                    separators=config.separators,
                    keep_separator=config.keep_separator,
                )
            return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name=config.encoding_name,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                separators=config.separators,
                keep_separator=config.keep_separator,
            )
        return RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            keep_separator=config.keep_separator,
        )

    @staticmethod
    def _ensure_tiktoken() -> None:
        """
        功能描述:
            校验当前运行环境是否安装 tiktoken 依赖。

        参数说明:
            无。

        返回值:
            None: 校验通过时无返回值。

        异常说明:
            ServiceException: 检测到缺失 tiktoken 依赖时抛出。
        """
        if importlib.util.find_spec("tiktoken") is None:
            raise ServiceException("递归切片使用 tiktoken 统计长度时需要安装 tiktoken")
