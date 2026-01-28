from pathlib import Path
from typing import List

from app.services.file_loader.base import FileLoader, PageContent


class TextLoader(FileLoader):
    """纯文本/Markdown 解析器，视为单页内容。"""

    def parse(self, file_path: Path) -> List[PageContent]:
        # 纯文本直接读取为单页
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        return [PageContent(page_number=1, text=text)]
