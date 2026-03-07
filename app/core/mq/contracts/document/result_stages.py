from __future__ import annotations

from enum import Enum


class DocumentChunkResultStage(str, Enum):
    """文档切片链路结果事件阶段枚举。

    Attributes:
        STARTED: 任务已接收，准备开始处理。
        COMPLETED: 切片相关处理已成功完成。
        FAILED: 任务处理失败。
    """

    STARTED = "STARTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ImportResultStage(str, Enum):
    """导入结果事件阶段枚举。

    Attributes:
        STARTED: 任务已接收，准备开始处理。
        PROCESSING: 任务处理中，表示导入主流程正在执行。
        COMPLETED: 导入流程已成功完成。
        FAILED: 导入流程执行失败。
    """

    STARTED = "STARTED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
