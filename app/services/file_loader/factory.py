from __future__ import annotations

from pathlib import Path
from typing import Dict, Type

from app.core.exceptions import ServiceException
from app.services.file_loader.base import FileLoader
from app.services.file_loader.docx_loader import WordLoader
from app.services.file_loader.excel_loader import ExcelLoader
from app.services.file_loader.html_loader import HtmlLoader
from app.services.file_loader.image_loader import ImageLoader
from app.services.file_loader.pdf_loader import PdfLoader
from app.services.file_loader.pptx_loader import PptxLoader
from app.services.file_loader.text_loader import TextLoader


class FileLoaderFactory:
    """文件解析器工厂，根据文件后缀返回对应解析器。"""

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
        # 根据后缀选择解析器
        suffix = file_path.suffix.lower()
        loader_cls = cls._loader_map.get(suffix)
        if not loader_cls:
            raise ServiceException(f"不支持的文件格式: {suffix}")
        return loader_cls()

    @classmethod
    def parse_file(cls, file_path: Path):
        loader = cls.get_loader(file_path)
        return loader.parse(file_path)
