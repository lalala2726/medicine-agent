from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

from app.rag.chunking.base import ChunkStrategyType, SplitChunk, SplitConfig
from app.rag.chunking.factory import ChunkerFactory
from app.rag.file_loader.factory import FileLoaderFactory


def split_file(
        file_path: Union[str, Path],
        strategy_type: str | ChunkStrategyType,
        config: Optional[SplitConfig] = None,
) -> List[SplitChunk]:
    """
    解析文件并按指定策略切片。

    Args:
        file_path: 文件路径
        strategy_type: 切片方式
        config: 切片配置

    Returns:
        切片结果列表
    """
    path = Path(file_path)
    pages = FileLoaderFactory.parse_file(path)
    resolved_config = config or SplitConfig()
    strategy = ChunkerFactory.get(strategy_type)
    return strategy.split_pages(pages, resolved_config)
