from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from app.services.file_loader.base import (
    FileLoader,
    ImageInfo,
    PageContent,
    copy_image_file,
    ensure_image_output_dir,
)


class ImageLoader(FileLoader):
    """图片文件解析器，仅标注图片信息，不做 OCR。"""

    def parse(
        self, file_path: Path, output_dir: Optional[Path] = None
    ) -> List[PageContent]:
        """
        解析图片文件：复制到输出目录并返回图片信息。

        Args:
            file_path: 图片文件路径
            output_dir: 图片输出目录（可选）

        Returns:
            单页内容（包含图片信息，无文本）
        """
        # 图片文件只做标注，不解析文本内容
        image_output_dir = ensure_image_output_dir(output_dir)
        # 复制图片到输出目录
        image_path = copy_image_file(
            output_dir=image_output_dir,
            source_path=file_path,
            file_stem=file_path.stem,
            page_number=1,
            index=1,
        )
        image_info = ImageInfo(index=1, name=file_path.name, path=str(image_path))
        return [PageContent(page_number=1, text="", images=[image_info])]
