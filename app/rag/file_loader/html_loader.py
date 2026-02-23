from html.parser import HTMLParser
from pathlib import Path
from typing import List, Optional

from app.rag.file_loader.base import FileLoader, PageContent


class _HTMLTextExtractor(HTMLParser):
    """简单 HTML 文本抽取器，用于将标签内容拼接为文本。"""

    def __init__(self) -> None:
        """初始化 HTML 解析器。"""
        super().__init__()
        self._parts: List[str] = []

    def handle_data(self, data: str) -> None:
        """
        处理 HTML 文本数据。

        Args:
            data: HTML 标签内的文本数据
        """
        if data and data.strip():
            self._parts.append(data.strip())

    def get_text(self) -> str:
        """
        获取合并后的文本内容。

        Returns:
            合并后的纯文本
        """
        return " ".join(" ".join(self._parts).split())


class HtmlLoader(FileLoader):
    """HTML 解析器，抽取正文文本。"""

    def parse(
            self, file_path: Path, output_dir: Optional[Path] = None
    ) -> List[PageContent]:
        """
        解析 HTML 文件，提取纯文本内容。

        Args:
            file_path: HTML 文件路径
            output_dir: 未使用（HTML 简单解析无图片）

        Returns:
            单页内容列表
        """
        # 简单抽取 HTML 文本为单页内容
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        extractor = _HTMLTextExtractor()
        extractor.feed(content)
        text = extractor.get_text()
        return [PageContent(page_number=1, text=text)]
