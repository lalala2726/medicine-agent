from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ImageInfo:
    """图片基础信息，用于标注页面中的图片。"""

    index: int
    name: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    mime_type: Optional[str] = None


@dataclass
class PageContent:
    """页面内容结构，统一输出格式。"""

    page_number: int
    text: str
    images: List[ImageInfo] = field(default_factory=list)
    page_label: Optional[str] = None

    def to_dict(self) -> dict:
        """转换为可序列化的字典结构，方便打印或存储。"""
        return {
            "page_number": self.page_number,
            "page_label": self.page_label,
            "text": self.text,
            "images": [asdict(image) for image in self.images],
            "has_images": bool(self.images),
            "image_count": len(self.images),
        }


class FileLoader(ABC):
    """文件解析器抽象基类，所有解析器需实现 parse。"""

    @abstractmethod
    def parse(self, file_path: Path) -> List[PageContent]:
        """解析文件并返回按页分组的内容。"""
        raise NotImplementedError
