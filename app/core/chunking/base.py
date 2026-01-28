from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from app.core.file_loader.base import PageContent


@dataclass
class SplitConfig:
    """
    文本切片配置。

    Attributes:
        chunk_size: 每个分块的目标大小
        chunk_overlap: 相邻分块之间的重叠长度
        separator: 基于字符分割时的分隔符
        separators: 递归分割时的分隔符列表
        is_separator_regex: 分隔符是否为正则表达式
        keep_separator: 是否保留分隔符
        use_tiktoken: 是否使用 tiktoken 统计长度
        encoding_name: tiktoken 编码名称（如 cl100k_base）
        model_name: 使用模型名称推断 tiktoken 编码
        headers_to_split_on: Markdown 标题层级配置
    """

    chunk_size: int = 1000
    chunk_overlap: int = 200
    separator: str = "\n\n"
    separators: Optional[List[str]] = None
    is_separator_regex: bool = False
    keep_separator: bool = False
    use_tiktoken: bool = False
    encoding_name: str = "cl100k_base"
    model_name: Optional[str] = None
    headers_to_split_on: Optional[List[Tuple[str, str]]] = None


class ChunkStrategyType(str, Enum):
    """切片策略枚举，统一对外传参。"""

    LENGTH = "length"
    TITLE = "title"
    TOKEN = "token"
    RECURSIVE = "recursive"


@dataclass
class SplitChunk:
    """
    切片后的文本块结构。

    Attributes:
        text: 分块文本内容
        page_number: 来源页码
        page_label: 来源页标签（如 Excel sheet 名称）
        chunk_index: 当前页内的分块序号（从 1 开始）
        metadata: 额外元数据（如标题、图片统计等）
    """

    text: str
    page_number: int
    page_label: Optional[str] = None
    chunk_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """转换为字典，便于日志输出或序列化。"""
        return {
            "text": self.text,
            "page_number": self.page_number,
            "page_label": self.page_label,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }


class ChunkStrategy(ABC):
    """切片策略抽象基类，支持按页处理。"""

    def split_pages(self, pages: List[PageContent], config: SplitConfig) -> List[SplitChunk]:
        """
        按页拆分文本，默认逐页调用 split_page。

        Args:
            pages: 页面内容列表
            config: 切片配置

        Returns:
            切片结果列表
        """
        results: List[SplitChunk] = []
        for page in pages:
            if not page.text:
                continue
            results.extend(self.split_page(page, config))
        return results

    @abstractmethod
    def split_page(self, page: PageContent, config: SplitConfig) -> List[SplitChunk]:
        """拆分单页文本。"""
        raise NotImplementedError


def build_page_metadata(page: PageContent) -> Dict[str, Any]:
    """构造基础页元数据。"""
    return {
        "page_number": page.page_number,
        "page_label": page.page_label,
        "has_images": bool(page.images),
        "image_count": len(page.images),
    }
