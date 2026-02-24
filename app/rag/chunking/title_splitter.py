from __future__ import annotations

from typing import List

from langchain_text_splitters import MarkdownHeaderTextSplitter

from app.rag.chunking.base import ChunkStrategy, SplitChunk, SplitConfig, build_page_metadata
from app.rag.file_loader.base import PageContent

DEFAULT_HEADERS = [
    ("#", "title"),
    ("##", "header"),
    ("###", "subheader"),
    ("####", "subheader2"),
]


class TitleChunker(ChunkStrategy):
    """基于 Markdown 标题结构的切片策略。"""

    def split_page(self, page: PageContent, config: SplitConfig) -> List[SplitChunk]:
        headers = config.headers_to_split_on or DEFAULT_HEADERS
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
        documents = splitter.split_text(page.text)
        base_metadata = build_page_metadata(page)
        chunks: List[SplitChunk] = []
        for index, doc in enumerate(documents):
            metadata = {**base_metadata, **doc.metadata}
            chunks.append(
                SplitChunk(
                    text=doc.page_content,
                    page_number=page.page_number,
                    page_label=page.page_label,
                    chunk_index=index + 1,
                    metadata=metadata,
                )
            )
        return chunks
