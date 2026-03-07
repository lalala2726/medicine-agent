"""切片重建流程结构化日志工具。

该模块只负责统一切片重建链路的日志阶段枚举和输出格式，不参与业务处理。
"""

from __future__ import annotations

from enum import Enum

from loguru import logger


class ChunkRebuildStage(str, Enum):
    """切片重建流程阶段枚举，覆盖命令消费与处理全链路。"""

    TASK_RECEIVED = "task_received"
    TASK_INVALID = "task_invalid"
    TASK_STALE_DROPPED = "task_stale_dropped"

    REBUILD_START = "rebuild_start"
    REBUILD_STALE = "rebuild_stale"
    REBUILD_FAILED = "rebuild_failed"

    COMPLETED = "completed"
    FAILED = "failed"

    RESULT_PUBLISHED = "result_published"
    RESULT_PUBLISH_FAILED = "result_publish_failed"

    CONSUMER_CONNECTED = "consumer_connected"
    CONSUMER_RECONNECTING = "consumer_reconnecting"


_ERROR_STAGES: frozenset[ChunkRebuildStage] = frozenset(
    {
        ChunkRebuildStage.FAILED,
        ChunkRebuildStage.RESULT_PUBLISH_FAILED,
        ChunkRebuildStage.TASK_INVALID,
        ChunkRebuildStage.REBUILD_FAILED,
    }
)

_WARNING_STAGES: frozenset[ChunkRebuildStage] = frozenset(
    {
        ChunkRebuildStage.CONSUMER_RECONNECTING,
        ChunkRebuildStage.TASK_STALE_DROPPED,
        ChunkRebuildStage.REBUILD_STALE,
    }
)


def chunk_rebuild_log(
    stage: ChunkRebuildStage, task_uuid: str = "-", /, **metrics: object
) -> None:
    """输出一条切片重建流程结构化日志。

    Args:
        stage: 切片重建流程阶段。
        task_uuid: 任务标识，未知时使用 "-"。
        **metrics: 附加键值对。
    """
    parts = " ".join(f"{k}={v}" for k, v in metrics.items()) if metrics else ""
    text = f"[chunk_rebuild] [task_uuid={task_uuid}] [{stage.value}] {parts}".rstrip()

    if stage in _ERROR_STAGES:
        logger.error(text)
    elif stage in _WARNING_STAGES:
        logger.warning(text)
    else:
        logger.info(text)
