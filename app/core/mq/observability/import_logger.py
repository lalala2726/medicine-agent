"""知识库导入流程结构化日志工具。

该模块只负责统一导入链路的日志阶段枚举和输出格式，不参与业务处理。
"""

from __future__ import annotations

from enum import Enum

from loguru import logger


class ImportStage(str, Enum):
    """导入流程阶段枚举，覆盖命令消费与处理全链路。

    Attributes:
        TASK_RECEIVED: 成功接收到合法任务。
        TASK_INVALID: 收到非法任务并丢弃。
        TASK_STALE_DROPPED: 收到旧版本任务并丢弃。
        DOWNLOAD_START: 开始下载文件。
        DOWNLOAD_DONE: 文件下载完成。
        PARSE_DONE: 文件解析完成。
        CHUNK_DONE: 文本切片完成。
        EMBED_BATCH: 单批次向量化执行中。
        EMBED_DONE: 全部向量化完成。
        INSERT_DONE: 向量写入完成。
        COMPLETED: 整体任务成功结束。
        FAILED: 整体任务失败结束。
        RESULT_PUBLISHED: 结果消息投递成功。
        RESULT_PUBLISH_FAILED: 结果消息投递失败。
        CONSUMER_CONNECTED: 消费者成功建立连接。
        CONSUMER_RECONNECTING: 消费者进入重连等待。
    """

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


# 需要按 error 级别输出的阶段集合。
_ERROR_STAGES: frozenset[ImportStage] = frozenset(
    {
        ImportStage.FAILED,
        ImportStage.RESULT_PUBLISH_FAILED,
        ImportStage.TASK_INVALID,
    }
)

# 需要按 warning 级别输出的阶段集合。
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
