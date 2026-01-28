from __future__ import annotations

from pathlib import Path
from typing import List

from app.services.file_loader.base import FileLoader, ImageInfo, PageContent


class ImageLoader(FileLoader):
    """图片文件解析器，仅标注图片信息，不做 OCR。"""

    def parse(self, file_path: Path) -> List[PageContent]:
        # 图片文件只做标注，不解析文本内容
        image_info = ImageInfo(index=1, name=file_path.name)
        return [PageContent(page_number=1, text="", images=[image_info])]
