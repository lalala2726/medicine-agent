from __future__ import annotations

import importlib.util
from typing import Any

from langchain_text_splitters import TokenTextSplitter

from app.core.exception.exceptions import ServiceException
from app.rag.chunking.types import (
    ChunkStrategy,
    SplitChunk,
    SplitConfig,
    build_chunk_stats,
)


class TokenChunker(ChunkStrategy):
    """
    功能描述:
        基于 token 数量的文本切片策略，用于严格控制每块 token 规模。

    参数说明:
        无。配置由 split_text 方法参数传入。

    返回值:
        无。通过 split_text 返回切片结果。

    异常说明:
        ServiceException: 当前环境未安装 tiktoken 依赖时抛出。
    """

    def split_text(self, text: str, config: SplitConfig) -> list[SplitChunk]:
        """
        功能描述:
            按 token 数量切分单段文本，返回标准切片结果。

        参数说明:
            text (str): 待切片文本。
            config (SplitConfig): 切片配置。

        返回值:
            list[SplitChunk]: 切片结果列表，每项包含文本和字符统计信息。

        异常说明:
            ServiceException: 缺失 tiktoken 依赖时抛出。
        """
        self._ensure_tiktoken()
        splitter_kwargs: dict[str, Any] = {
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
        }
        if config.model_name:
            splitter_kwargs["model_name"] = config.model_name
        elif config.encoding_name:
            splitter_kwargs["encoding_name"] = config.encoding_name
        # TODO(zhangchuang): 离线环境下 tiktoken 可能首次下载编码文件失败，
        # 需补充本地缓存预热或降级策略，避免网络依赖导致切片失败。
        splitter = TokenTextSplitter(**splitter_kwargs)
        pieces = splitter.split_text(text)
        return [
            SplitChunk(
                text=piece,
                stats=build_chunk_stats(piece),
            )
            for piece in pieces
        ]

    @staticmethod
    def _ensure_tiktoken() -> None:
        """
        功能描述:
            校验当前运行环境是否安装 tiktoken 依赖。

        参数说明:
            无。

        返回值:
            None: 校验通过时无返回值。

        异常说明:
            ServiceException: 检测到缺失 tiktoken 依赖时抛出。
        """
        if importlib.util.find_spec("tiktoken") is None:
            raise ServiceException("TokenTextSplitter 依赖 tiktoken，请先安装")
