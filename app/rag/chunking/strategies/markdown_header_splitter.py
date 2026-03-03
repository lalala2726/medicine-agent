from __future__ import annotations

from langchain_text_splitters import MarkdownHeaderTextSplitter

from app.rag.chunking.types import ChunkStrategy, SplitChunk, SplitConfig

DEFAULT_HEADERS: list[tuple[str, str]] = [
    ("#", "title"),
    ("##", "header"),
    ("###", "subheader"),
    ("####", "subheader2"),
]


class MarkdownHeaderChunker(ChunkStrategy):
    """
    功能描述:
        基于 Markdown 标题结构进行切片，并携带标题层级元数据。

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
            按 Markdown 标题分层切分文本，并将 LangChain 返回的标题元数据写入 chunk.metadata。

        参数说明:
            text (str): 待切片文本。
            config (SplitConfig): 切片配置。

        返回值:
            list[SplitChunk]: 切片结果列表，包含标题元数据，chunk_index 从 1 递增。

        异常说明:
            无。
        """
        headers = config.headers_to_split_on or DEFAULT_HEADERS
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
        documents = splitter.split_text(text)
        return [
            SplitChunk(
                text=document.page_content,
                page_number=1,
                page_label=None,
                chunk_index=index + 1,
                metadata={**document.metadata},
            )
            for index, document in enumerate(documents)
        ]
