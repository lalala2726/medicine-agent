"""手工新增切片流程结构化日志工具。

该模块只负责统一手工新增切片链路的日志阶段枚举和输出格式，不参与业务处理。
"""

from __future__ import annotations

from enum import Enum

from loguru import logger


class ChunkAddStage(str, Enum):
    """手工新增切片流程阶段枚举，覆盖命令消费与处理全链路。"""

    TASK_RECEIVED = "task_received"
    TASK_INVALID = "task_invalid"

    ADD_START = "add_start"
    ADD_FAILED = "add_failed"

    COMPLETED = "completed"
    FAILED = "failed"

    RESULT_PUBLISHED = "result_published"
    RESULT_PUBLISH_FAILED = "result_publish_failed"

    CONSUMER_CONNECTED = "consumer_connected"
    CONSUMER_RECONNECTING = "consumer_reconnecting"


_ERROR_STAGES: frozenset[ChunkAddStage] = frozenset(
    {
        ChunkAddStage.FAILED,
        ChunkAddStage.RESULT_PUBLISH_FAILED,
        ChunkAddStage.TASK_INVALID,
        ChunkAddStage.ADD_FAILED,
    }
)

_WARNING_STAGES: frozenset[ChunkAddStage] = frozenset(
    {
        ChunkAddStage.CONSUMER_RECONNECTING,
    }
)


def chunk_add_log(
    stage: ChunkAddStage, task_uuid: str = "-", /, **metrics: object
) -> None:
    """输出一条手工新增切片流程结构化日志。

    Args:
        stage: 手工新增切片流程阶段。
        task_uuid: 任务标识，未知时使用 "-"。
        **metrics: 附加键值对。
    """
    parts = " ".join(f"{k}={v}" for k, v in metrics.items()) if metrics else ""
    text = f"[chunk_add] [task_uuid={task_uuid}] [{stage.value}] {parts}".rstrip()

    if stage in _ERROR_STAGES:
        logger.error(text)
    elif stage in _WARNING_STAGES:
        logger.warning(text)
    else:
        logger.info(text)
