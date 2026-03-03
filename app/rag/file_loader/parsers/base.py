from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from app.rag.file_loader.types import ParsedPage


class BaseParser(ABC):
    """
    功能描述:
        定义文件解析器抽象基类，约束不同文件类型解析器的统一输入输出协议。

    参数说明:
        无。子类需实现 `parse` 方法。

    返回值:
        无。通过 `parse` 返回结构化页面列表。

    异常说明:
        无。异常由具体解析器实现抛出。
    """

    @abstractmethod
    def parse(self, file_path: Path) -> list[ParsedPage]:
        """
        功能描述:
            解析指定文件并返回页面文本列表。

        参数说明:
            file_path (Path): 待解析文件路径。

        返回值:
            list[ParsedPage]: 按页面组织的解析结果。

        异常说明:
            ServiceException: 文件格式不支持、依赖缺失或解析失败时由子类抛出。
        """
        raise NotImplementedError
