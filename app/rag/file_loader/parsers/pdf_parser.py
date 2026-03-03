from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from app.rag.file_loader.parsers.base import BaseParser
from app.rag.file_loader.types import ParsedPage


class PdfParser(BaseParser):
    """
    功能描述:
        解析 PDF 文件并按页提取文本内容。

    参数说明:
        无。解析参数通过 `parse` 方法传入。

    返回值:
        无。调用 `parse` 时返回页面列表。

    异常说明:
        无。底层解析异常由 pypdf 抛出。
    """

    def parse(self, file_path: Path) -> list[ParsedPage]:
        """
        功能描述:
            打开 PDF 文件并按页提取文本。

        参数说明:
            file_path (Path): PDF 文件路径。

        返回值:
            list[ParsedPage]: 按页组织的文本列表。

        异常说明:
            Exception: PDF 结构损坏或读取失败时由底层库抛出。
        """
        reader = PdfReader(str(file_path))
        pages: list[ParsedPage] = []
        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append(ParsedPage(page_number=index, text=text))
        return pages
