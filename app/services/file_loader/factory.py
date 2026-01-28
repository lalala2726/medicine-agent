from __future__ import annotations

from pathlib import Path
from typing import Dict, Type

from app.core.codes import ResponseCode
from app.core.exceptions import ServiceException
from app.services.file_loader.base import (
    FileLoader,
    create_temp_image_dir,
    register_temp_assets,
)
from app.services.file_loader.docx_loader import WordLoader
from app.services.file_loader.excel_loader import ExcelLoader
from app.services.file_loader.html_loader import HtmlLoader
from app.services.file_loader.image_loader import ImageLoader
from app.services.file_loader.pdf_loader import PdfLoader
from app.services.file_loader.pptx_loader import PptxLoader
from app.services.file_loader.text_loader import TextLoader


class FileLoaderFactory:
    """文件解析器工厂，根据文件后缀返回对应解析器。"""

    # 文件扩展名到解析器类的映射表
    _loader_map: Dict[str, Type[FileLoader]] = {
        ".txt": TextLoader,
        ".md": TextLoader,
        ".html": HtmlLoader,
        ".htm": HtmlLoader,
        ".pdf": PdfLoader,
        ".docx": WordLoader,
        ".doc": WordLoader,
        ".pptx": PptxLoader,
        ".ppt": PptxLoader,
        ".xlsx": ExcelLoader,
        ".xls": ExcelLoader,
        ".png": ImageLoader,
        ".jpg": ImageLoader,
        ".jpeg": ImageLoader,
        ".bmp": ImageLoader,
        ".gif": ImageLoader,
        ".tif": ImageLoader,
        ".tiff": ImageLoader,
    }

    @classmethod
    def get_loader(cls, file_path: Path) -> FileLoader:
        """
        根据文件后缀获取对应的解析器实例。

        Args:
            file_path: 文件路径

        Returns:
            对应的文件解析器实例

        Raises:
            ServiceException: 不支持的文件格式
        """
        # 根据后缀选择解析器
        suffix = file_path.suffix.lower()
        loader_cls = cls._loader_map.get(suffix)
        if not loader_cls:
            raise ServiceException(
                code=ResponseCode.BAD_REQUEST, message=f"不支持的文件格式: {suffix}"
            )
        return loader_cls()

    @classmethod
    def parse_file(cls, file_path: Path, output_dir: Path | None = None):
        """
        解析文件并返回页面内容列表。

        Args:
            file_path: 文件路径
            output_dir: 图片输出目录（可选）

        Returns:
            页面内容列表
        """
        loader = cls.get_loader(file_path)
        return loader.parse(file_path, output_dir=output_dir)

    @classmethod
    def parse_file_with_images(
            cls, file_path: Path, source_name: str | None = None
    ) -> dict:
        """
        解析文件并提取图片到临时目录，返回结构化结果。

        Args:
            file_path: 文件路径
            source_name: 源文件名（用于注册清理资源）

        Returns:
            包含图片目录和页面内容的字典
        """
        # 在文件解析层创建临时目录并返回结构化结果
        file_stem = file_path.stem or "file"
        image_dir = create_temp_image_dir(file_stem)
        loader = cls.get_loader(file_path)
        pages = loader.parse(file_path, output_dir=image_dir)
        # 注册临时资源以便后续清理
        register_temp_assets(
            source_name or file_path.name, image_dir, source_path=file_path
        )
        return {
            "image_dir": str(image_dir),
            "pages": [page.to_dict() for page in pages],
        }
