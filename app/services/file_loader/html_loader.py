from html.parser import HTMLParser
from pathlib import Path
from typing import List

from app.services.file_loader.base import FileLoader, PageContent


class _HTMLTextExtractor(HTMLParser):
    """简单 HTML 文本抽取器，用于将标签内容拼接为文本。"""

    def __init__(self) -> None:
        super().__init__()
        self._parts: List[str] = []

    def handle_data(self, data: str) -> None:
        if data and data.strip():
            self._parts.append(data.strip())

    def get_text(self) -> str:
        return " ".join(" ".join(self._parts).split())


class HtmlLoader(FileLoader):
    """HTML 解析器，抽取正文文本。"""

    def parse(self, file_path: Path) -> List[PageContent]:
        # 简单抽取 HTML 文本为单页内容
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        extractor = _HTMLTextExtractor()
        extractor.feed(content)
        text = extractor.get_text()
        return [PageContent(page_number=1, text=text)]
