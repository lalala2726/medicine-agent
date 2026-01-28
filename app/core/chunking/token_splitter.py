from __future__ import annotations

import importlib.util
from typing import List

from langchain_text_splitters import TokenTextSplitter

from app.core.exceptions import ServiceException
from app.core.chunking.base import ChunkStrategy, SplitChunk, SplitConfig, build_page_metadata
from app.core.file_loader.base import PageContent


class TokenChunker(ChunkStrategy):
    """基于 token 数量的切片策略（严格控制 token 数量）。"""

    def split_page(self, page: PageContent, config: SplitConfig) -> List[SplitChunk]:
        self._ensure_tiktoken()
        splitter_kwargs = {
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
        }
        if config.model_name:
            splitter_kwargs["model_name"] = config.model_name
        elif config.encoding_name:
            splitter_kwargs["encoding_name"] = config.encoding_name
        splitter = TokenTextSplitter(**splitter_kwargs)
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

    @staticmethod
    def _ensure_tiktoken() -> None:
        if importlib.util.find_spec("tiktoken") is None:
            raise ServiceException("TokenTextSplitter 依赖 tiktoken，请先安装")
