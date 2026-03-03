from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
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


@dataclass
class ChunkStats:
    """
    功能描述:
        定义单个切片的文本统计信息结构。

    参数说明:
        char_count (int): 切片文本字符数量，默认值为 0。

    返回值:
        无。该类用于承载切片统计数据。

    异常说明:
        无。
    """

    char_count: int = 0

    def to_dict(self) -> dict[str, int]:
        """
        功能描述:
            将统计对象转换为可序列化字典。

        参数说明:
            无。

        返回值:
            dict[str, int]: 仅包含 `char_count` 字段的字典。

        异常说明:
            无。
        """
        return {
            "char_count": self.char_count,
        }


def build_chunk_stats(text: str) -> ChunkStats:
    """
    功能描述:
        基于切片文本构建统计信息对象。

    参数说明:
        text (str): 切片文本内容。

    返回值:
        ChunkStats: 包含字符统计信息的对象。

    异常说明:
        无。空文本会返回 `char_count=0`。
    """
    return ChunkStats(char_count=len(text or ""))


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
        stats (ChunkStats): 切片统计信息，默认基于 text 自动计算。

    返回值:
        无。该类用于承载切片结果数据。

    异常说明:
        无。字段完整性由调用方保障。
    """

    text: str
    stats: ChunkStats

    def to_dict(self) -> dict[str, Any]:
        """
        功能描述:
            将切片对象转换为字典结构，便于序列化与日志输出。

        参数说明:
            无。

        返回值:
            dict[str, Any]: 包含文本和统计信息的字典。

        异常说明:
            无。
        """
        return {
            "text": self.text,
            "stats": self.stats.to_dict(),
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
