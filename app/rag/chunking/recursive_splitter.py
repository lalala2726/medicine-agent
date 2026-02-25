from __future__ import annotations

import importlib.util
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.exception.exceptions import ServiceException
from app.rag.chunking.base import ChunkStrategy, SplitChunk, SplitConfig, build_page_metadata
from app.rag.file_loader.base import PageContent


class RecursiveChunker(ChunkStrategy):
    """基于文本结构递归切片策略（优先保留段落/句子）。"""

    def split_page(self, page: PageContent, config: SplitConfig) -> List[SplitChunk]:
        splitter = self._build_splitter(config)
        pieces = splitter.split_text(page.text)
        base_metadata = build_page_metadata(page)
        return [
            SplitChunk(
                text=piece,
                page_number=page.page_number,
                page_label=page.page_label,
                chunk_index=index + 1,
                metadata={**base_metadata},
            )
            for index, piece in enumerate(pieces)
        ]

    def _build_splitter(self, config: SplitConfig) -> RecursiveCharacterTextSplitter:
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
        if importlib.util.find_spec("tiktoken") is None:
            raise ServiceException("递归切片使用 tiktoken 统计长度时需要安装 tiktoken")
