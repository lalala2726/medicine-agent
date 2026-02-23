from __future__ import annotations

import importlib.util
from typing import List

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from app.rag.chunking.base import ChunkStrategy, SplitChunk, SplitConfig, build_page_metadata
from app.core.exception.exceptions import ServiceException
from app.rag.file_loader.base import PageContent


class LengthChunker(ChunkStrategy):
    """基于长度的切片策略（按字符长度，支持 tiktoken 计数）。"""

    def split_page(self, page: PageContent, config: SplitConfig) -> List[SplitChunk]:
        splitter = self._build_splitter(config)
        pieces = splitter.split_text(page.text)
        # 如果字符分割无法满足块大小，使用递归分割确保不超限
        if pieces and max(len(piece) for piece in pieces) > config.chunk_size:
            pieces = self._build_fallback_splitter(config).split_text(page.text)
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

    def _build_splitter(self, config: SplitConfig) -> CharacterTextSplitter:
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

    def _build_fallback_splitter(self, config: SplitConfig) -> RecursiveCharacterTextSplitter:
        separators = config.separators or ["\\n\\n", "\\n", " ", ""]
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
        if importlib.util.find_spec("tiktoken") is None:
            raise ServiceException("按 token 长度切片需要安装 tiktoken")
