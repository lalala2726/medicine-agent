from pathlib import Path
from typing import List, Optional

from app.services.file_loader.base import FileLoader, PageContent


class TextLoader(FileLoader):
    """纯文本/Markdown 解析器，视为单页内容。"""

    def parse(
        self, file_path: Path, output_dir: Optional[Path] = None
    ) -> List[PageContent]:
        """
        解析纯文本或 Markdown 文件。

        Args:
            file_path: 文件路径
            output_dir: 未使用（纯文本无图片）

        Returns:
            单页内容列表
        """
        # 纯文本直接读取为单页
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        return [PageContent(page_number=1, text=text)]
