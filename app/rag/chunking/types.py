from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass
class SplitConfig:
    """
    功能描述:
        定义文本切片所需的通用配置参数，供不同切片策略统一读取。

    参数说明:
        chunk_size (int): 每个分块的目标大小，默认值为 1000。
        chunk_overlap (int): 相邻分块重叠大小，默认值为 200。
        separator (str): 字符切片时的主分隔符，默认值为 "\\n\\n"。
        separators (list[str] | None): 递归切片的分隔符列表，默认值为 None。
        is_separator_regex (bool): separator 是否按正则表达式解析，默认值为 False。
        keep_separator (bool): 是否在切片结果中保留分隔符，默认值为 False。
        use_tiktoken (bool): 是否使用 tiktoken 计算长度，默认值为 False。
        encoding_name (str): tiktoken 编码名称，默认值为 "cl100k_base"。
        model_name (str | None): 用于推断 tiktoken 编码的模型名，默认值为 None。
        headers_to_split_on (list[tuple[str, str]] | None): Markdown 标题切片规则，默认值为 None。

    返回值:
        无。该类用于承载切片配置数据。

    异常说明:
        无。参数合法性由调用方或具体策略在运行时校验。
    """

    chunk_size: int = 1000
    chunk_overlap: int = 200
    separator: str = "\n\n"
    separators: list[str] | None = None
    is_separator_regex: bool = False
    keep_separator: bool = False
    use_tiktoken: bool = False
    encoding_name: str = "cl100k_base"
    model_name: str | None = None
    headers_to_split_on: list[tuple[str, str]] | None = None


class ChunkStrategyType(str, Enum):
    """
    功能描述:
        定义对外可用的切片策略类型枚举，统一 API 入参与内部工厂映射。

    参数说明:
        无。枚举项固定在类定义中。

    返回值:
        无。调用方通过枚举成员的 value 获取传输值。

    异常说明:
        无。非法值在参数绑定或策略获取阶段抛出异常。
    """

    CHARACTER = "character"
    RECURSIVE = "recursive"
    TOKEN = "token"
    MARKDOWN_HEADER = "markdown_header"


@dataclass
class SplitChunk:
    """
    功能描述:
        表示单个切片结果，统一输出给上层业务使用。

    参数说明:
        text (str): 切片文本内容。
        page_number (int): 来源页码，纯文本默认值为 1。
        page_label (str | None): 来源页标签，纯文本默认值为 None。
        chunk_index (int): 当前页内切片序号，从 1 开始。
        metadata (dict[str, Any]): 额外元数据，默认值为空字典。

    返回值:
        无。该类用于承载切片结果数据。

    异常说明:
        无。字段完整性由调用方保障。
    """

    text: str
    page_number: int = 1
    page_label: str | None = None
    chunk_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        功能描述:
            将切片对象转换为字典结构，便于序列化与日志输出。

        参数说明:
            无。

        返回值:
            dict[str, Any]: 包含文本、页码、序号和元数据的字典。

        异常说明:
            无。
        """
        return {
            "text": self.text,
            "page_number": self.page_number,
            "page_label": self.page_label,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }


class ChunkStrategy(ABC):
    """
    功能描述:
        定义切片策略抽象基类，约束各策略的统一输入输出协议。

    参数说明:
        无。

    返回值:
        无。子类需实现 split_text 方法。

    异常说明:
        无。异常由具体策略实现抛出。
    """

    @abstractmethod
    def split_text(self, text: str, config: SplitConfig) -> list[SplitChunk]:
        """
        功能描述:
            对单段文本执行策略切片并返回结构化切片结果。

        参数说明:
            text (str): 待切片文本。
            config (SplitConfig): 切片配置。

        返回值:
            list[SplitChunk]: 切片结果列表。

        异常说明:
            NotImplementedError: 子类未实现该方法时抛出。
        """
        raise NotImplementedError
