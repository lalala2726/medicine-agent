"""
知识库导入流程结构化日志模块。

设计原则：
- 业务代码仅需一行调用即可输出完整结构化日志。
- 日志级别由 ImportStage 枚举自动推断，无需调用方关心。
- 日志格式统一为 ``[import] [task_uuid=xxx] [stage] key=value ...``。
"""

from __future__ import annotations

from enum import Enum

from loguru import logger


class ImportStage(str, Enum):
    """导入流程关键阶段枚举，覆盖全链路节点。"""

    # ── 消息生命周期 ──
    TASK_RECEIVED = "task_received"
    TASK_INVALID = "task_invalid"

    # ── 处理流程 ──
    DOWNLOAD_START = "download_start"
    DOWNLOAD_DONE = "download_done"
    PARSE_DONE = "parse_done"
    CHUNK_DONE = "chunk_done"
    EMBED_BATCH = "embed_batch"
    EMBED_DONE = "embed_done"
    INSERT_DONE = "insert_done"

    # ── 终态 ──
    COMPLETED = "completed"
    FAILED = "failed"

    # ── 回调 ──
    CALLBACK_SENT = "callback_sent"
    CALLBACK_FAILED = "callback_failed"
    CALLBACK_SKIPPED = "callback_skipped"

    # ── 重试 ──
    RETRY_SCHEDULED = "retry_scheduled"

    # ── 消费者连接 ──
    CONSUMER_CONNECTED = "consumer_connected"
    CONSUMER_RECONNECTING = "consumer_reconnecting"


# 使用 error 级别的阶段集合
_ERROR_STAGES: frozenset[ImportStage] = frozenset(
    {
        ImportStage.FAILED,
        ImportStage.CALLBACK_FAILED,
        ImportStage.TASK_INVALID,
    }
)

# 使用 warning 级别的阶段集合
_WARNING_STAGES: frozenset[ImportStage] = frozenset(
    {
        ImportStage.RETRY_SCHEDULED,
        ImportStage.CALLBACK_SKIPPED,
        ImportStage.CONSUMER_RECONNECTING,
    }
)


def import_log(
    stage: ImportStage,
    task_uuid: str = "-",
    /,
    **metrics: object,
) -> None:
    """
    输出一条导入流程结构化日志。

    Args:
        stage: 当前阶段枚举值。
        task_uuid: 导入任务唯一标识，未知时传 ``"-"``。
        **metrics: 附加指标键值对（如 ``filename="a.pdf"``, ``duration_ms=320``）。

    Returns:
        None。

    Raises:
        无。日志输出不会抛出异常。

    Usage::

        import_log(ImportStage.DOWNLOAD_DONE, task_uuid, filename="a.pdf", size=10240)
    """
    parts = " ".join(f"{k}={v}" for k, v in metrics.items()) if metrics else ""
    text = f"[import] [task_uuid={task_uuid}] [{stage.value}] {parts}".rstrip()

    if stage in _ERROR_STAGES:
        logger.error(text)
    elif stage in _WARNING_STAGES:
        logger.warning(text)
    else:
        logger.info(text)
