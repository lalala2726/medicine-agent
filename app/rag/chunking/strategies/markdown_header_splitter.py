from __future__ import annotations

from langchain_text_splitters import MarkdownHeaderTextSplitter

from app.rag.chunking.types import (
    ChunkStrategy,
    SplitChunk,
    SplitConfig,
    build_chunk_stats,
)

DEFAULT_HEADERS: list[tuple[str, str]] = [
    ("#", "title"),
    ("##", "header"),
    ("###", "subheader"),
    ("####", "subheader2"),
]


class MarkdownHeaderChunker(ChunkStrategy):
    """
    功能描述:
        基于 Markdown 标题结构进行切片。

    参数说明:
        无。配置由 split_text 方法参数传入。

    返回值:
        无。通过 split_text 返回切片结果。

    异常说明:
        无。
    """

    def split_text(self, text: str, config: SplitConfig) -> list[SplitChunk]:
        """
        功能描述:
            按 Markdown 标题分层切分文本。

        参数说明:
            text (str): 待切片文本。
            config (SplitConfig): 切片配置。

        返回值:
            list[SplitChunk]: 切片结果列表，每项包含文本和字符统计信息。

        异常说明:
            无。
        """
        headers = config.headers_to_split_on or DEFAULT_HEADERS
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
        documents = splitter.split_text(text)
        return [
            SplitChunk(
                text=document.page_content,
                stats=build_chunk_stats(document.page_content),
            )
            for document in documents
        ]
