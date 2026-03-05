"""知识库导入流程结构化日志工具。"""

from __future__ import annotations

from enum import Enum

from loguru import logger


class ImportStage(str, Enum):
    """导入流程阶段枚举，覆盖命令消费与处理全链路。"""

    TASK_RECEIVED = "task_received"
    TASK_INVALID = "task_invalid"
    TASK_STALE_DROPPED = "task_stale_dropped"

    DOWNLOAD_START = "download_start"
    DOWNLOAD_DONE = "download_done"
    PARSE_DONE = "parse_done"
    CHUNK_DONE = "chunk_done"
    EMBED_BATCH = "embed_batch"
    EMBED_DONE = "embed_done"
    INSERT_DONE = "insert_done"

    COMPLETED = "completed"
    FAILED = "failed"

    RESULT_PUBLISHED = "result_published"
    RESULT_PUBLISH_FAILED = "result_publish_failed"

    CONSUMER_CONNECTED = "consumer_connected"
    CONSUMER_RECONNECTING = "consumer_reconnecting"


_ERROR_STAGES: frozenset[ImportStage] = frozenset(
    {
        ImportStage.FAILED,
        ImportStage.RESULT_PUBLISH_FAILED,
        ImportStage.TASK_INVALID,
    }
)

_WARNING_STAGES: frozenset[ImportStage] = frozenset(
    {
        ImportStage.CONSUMER_RECONNECTING,
        ImportStage.TASK_STALE_DROPPED,
    }
)


def import_log(stage: ImportStage, task_uuid: str = "-", /, **metrics: object) -> None:
    """输出一条导入流程结构化日志。

    Args:
        stage: 导入流程阶段。
        task_uuid: 任务标识，未知时使用 "-"。
        **metrics: 附加键值对。

    Returns:
        None。
    """
    parts = " ".join(f"{k}={v}" for k, v in metrics.items()) if metrics else ""
    text = f"[import] [task_uuid={task_uuid}] [{stage.value}] {parts}".rstrip()

    if stage in _ERROR_STAGES:
        logger.error(text)
    elif stage in _WARNING_STAGES:
        logger.warning(text)
    else:
        logger.info(text)
