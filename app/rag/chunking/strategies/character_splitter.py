from __future__ import annotations

import importlib.util

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from app.core.exception.exceptions import ServiceException
from app.rag.chunking.types import ChunkStrategy, SplitChunk, SplitConfig


class CharacterChunker(ChunkStrategy):
    """
    功能描述:
        基于字符长度的文本切片策略，支持普通字符长度和 tiktoken 计数两种模式。

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
            对输入文本执行字符长度切片；若主切片器结果超出上限，自动降级到递归切片。

        参数说明:
            text (str): 待切片文本。
            config (SplitConfig): 切片配置。

        返回值:
            list[SplitChunk]: 切片结果列表，page_number 默认 1，chunk_index 从 1 递增。

        异常说明:
            ServiceException: 当 use_tiktoken=True 且未安装 tiktoken 时抛出。
        """
        splitter = self._build_splitter(config)
        pieces = splitter.split_text(text)
        if pieces and max(len(piece) for piece in pieces) > config.chunk_size:
            pieces = self._build_fallback_splitter(config).split_text(text)
        return [
            SplitChunk(
                text=piece,
                page_number=1,
                page_label=None,
                chunk_index=index + 1,
                metadata={},
            )
            for index, piece in enumerate(pieces)
        ]

    def _build_splitter(self, config: SplitConfig) -> CharacterTextSplitter:
        """
        功能描述:
            根据配置构建主字符切片器实例。

        参数说明:
            config (SplitConfig): 切片配置。

        返回值:
            CharacterTextSplitter: 主字符切片器实例。

        异常说明:
            ServiceException: 当 use_tiktoken=True 且未安装 tiktoken 时抛出。
        """
        if config.use_tiktoken:
            self._ensure_tiktoken()
            if config.model_name:
                return CharacterTextSplitter.from_tiktoken_encoder(
                    model_name=config.model_name,
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                    separator=config.separator,
                    is_separator_regex=config.is_separator_regex,
                )
            return CharacterTextSplitter.from_tiktoken_encoder(
                encoding_name=config.encoding_name,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                separator=config.separator,
                is_separator_regex=config.is_separator_regex,
            )
        return CharacterTextSplitter(
            separator=config.separator,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            is_separator_regex=config.is_separator_regex,
        )

    def _build_fallback_splitter(
        self,
        config: SplitConfig,
    ) -> RecursiveCharacterTextSplitter:
        """
        功能描述:
            构建递归切片器作为字符切片降级兜底，确保切片长度不超过配置上限。

        参数说明:
            config (SplitConfig): 切片配置。

        返回值:
            RecursiveCharacterTextSplitter: 递归切片器实例。

        异常说明:
            ServiceException: 当 use_tiktoken=True 且未安装 tiktoken 时抛出。
        """
        separators = config.separators or ["\n\n", "\n", " ", ""]
        if config.use_tiktoken:
            self._ensure_tiktoken()
            if config.model_name:
                return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                    model_name=config.model_name,
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                    separators=separators,
                    keep_separator=config.keep_separator,
                )
            return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name=config.encoding_name,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                separators=separators,
                keep_separator=config.keep_separator,
            )
        return RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=separators,
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
            raise ServiceException("按 token 长度切片需要安装 tiktoken")
