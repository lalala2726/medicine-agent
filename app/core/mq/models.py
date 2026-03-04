from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field

from app.rag.chunking import ChunkStrategyType


class CallbackStage(str, Enum):
    """
    功能描述:
        导入回调阶段枚举，表示任务在生命周期中的位置。

    枚举值:
        STARTED: 任务已接收，即将开始处理。
        PROCESSING: 任务正在处理中（下载/解析/切片/向量化）。
        COMPLETED: 任务全部完成。
        FAILED: 任务失败。
    """

    STARTED = "STARTED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

    # ── 向后兼容映射 ──

    @classmethod
    def from_legacy(cls, status: str) -> "CallbackStage":
        """
        功能描述:
            将旧版 SUCCESS/FAILED 状态映射到新枚举值。

        参数说明:
            status (str): 旧状态字符串。

        返回值:
            CallbackStage: 对应的枚举值。

        异常说明:
            ValueError: 无法识别的状态字符串。
        """
        mapping = {"SUCCESS": cls.COMPLETED, "FAILED": cls.FAILED}
        if status in mapping:
            return mapping[status]
        return cls(status)


class KnowledgeImportMessage(BaseModel):
    """
    功能描述:
        定义知识库导入任务的 MQ 消息结构。

    参数说明:
        task_uuid (str): 任务唯一标识。
        knowledge_name (str): 知识库名称。
        document_id (int): 文档 ID。
        file_url (str): 单个文件 URL。
        embedding_model (str): 向量模型名称（由 SpringBoot 透传）。
        chunk_strategy (ChunkStrategyType): 切片策略类型。
        chunk_size (int): 字符切片大小。
        token_size (int): token 切片大小。
        created_at (datetime): 消息创建时间。

    返回值:
        无。该类用于承载消息体数据。

    异常说明:
        无。字段校验由 Pydantic 自动处理。
    """

    task_uuid: str = Field(..., min_length=1)
    knowledge_name: str = Field(..., min_length=1)
    document_id: int = Field(..., gt=0)
    file_url: str = Field(..., min_length=1)
    embedding_model: str = Field(..., min_length=1)
    chunk_strategy: ChunkStrategyType = Field(default=ChunkStrategyType.CHARACTER)
    chunk_size: int = Field(default=500, ge=1)
    token_size: int = Field(default=100, ge=1)
    created_at: datetime

    @classmethod
    def build(
            cls,
            *,
            knowledge_name: str,
            document_id: int,
            file_url: str,
            embedding_model: str,
            chunk_strategy: ChunkStrategyType,
            chunk_size: int,
            token_size: int,
    ) -> "KnowledgeImportMessage":
        """
        功能描述:
            构建标准导入消息对象并自动填充 task_uuid 与创建时间。

        参数说明:
            knowledge_name (str): 知识库名称。
            document_id (int): 文档 ID。
            file_url (str): 单个文件 URL。
            embedding_model (str): 向量模型名称。
            chunk_strategy (ChunkStrategyType): 切片策略。
            chunk_size (int): 字符切片大小。
            token_size (int): token 切片大小。

        返回值:
            KnowledgeImportMessage: 完整消息对象。

        异常说明:
            无。字段非法时由 Pydantic 抛出验证异常。
        """
        return cls(
            task_uuid=str(uuid4()),
            knowledge_name=knowledge_name,
            document_id=document_id,
            file_url=file_url,
            embedding_model=embedding_model,
            chunk_strategy=chunk_strategy,
            chunk_size=chunk_size,
            token_size=token_size,
            created_at=datetime.now(timezone.utc),
        )

    def to_json_bytes(self) -> bytes:
        """
        功能描述:
            将消息模型序列化为 UTF-8 JSON 字节串，供 RabbitMQ 发布使用。

        参数说明:
            无。

        返回值:
            bytes: JSON 编码后的消息体字节串。

        异常说明:
            无。序列化失败时由 Pydantic 运行时抛出异常。
        """
        return self.model_dump_json().encode("utf-8")


class KnowledgeImportCallbackPayload(BaseModel):
    """
    功能描述:
        定义知识库导入回调参数结构，支持多阶段回调（STARTED / PROCESSING / COMPLETED / FAILED）。

    参数说明:
        task_uuid (str): 导入任务唯一标识。
        knowledge_name (str): 知识库名称。
        document_id (int): 文档 ID。
        file_url (str): 导入文件 URL。
        status (str): 回调阶段，取值 STARTED / PROCESSING / COMPLETED / FAILED。
        stage_detail (str | None): 可选子阶段描述（如 "downloading"、"embedding batch 3/5"）。
        message (str): 结果摘要或失败原因。
        embedding_model (str): 向量模型名称。
        embedding_dim (int): 向量维度（从 Milvus schema 读取）。
        chunk_strategy (str): 切片策略。
        chunk_size (int): 字符切片大小。
        token_size (int): token 切片大小。
        chunk_count (int): 切片总数。
        vector_count (int): 写入向量总数。
        started_at (datetime): 任务开始时间（UTC）。
        finished_at (datetime): 任务结束时间（UTC）。
        duration_ms (int): 处理耗时（毫秒）。

    返回值:
        无。该类用于承载回调参数数据。

    异常说明:
        无。字段校验由 Pydantic 自动处理。
    """

    task_uuid: str = Field(..., min_length=1)
    knowledge_name: str = Field(..., min_length=1)
    document_id: int = Field(..., gt=0)
    file_url: str = Field(..., min_length=1)
    status: str = Field(..., min_length=1)
    stage_detail: str | None = Field(default=None)
    message: str = Field(..., min_length=1)
    embedding_model: str = Field(..., min_length=1)
    embedding_dim: int = Field(..., ge=0)
    chunk_strategy: str = Field(..., min_length=1)
    chunk_size: int = Field(..., ge=1)
    token_size: int = Field(..., ge=1)
    chunk_count: int = Field(default=0, ge=0)
    vector_count: int = Field(default=0, ge=0)
    started_at: datetime
    finished_at: datetime
    duration_ms: int = Field(..., ge=0)

    @classmethod
    def build(
            cls,
            *,
            task_uuid: str,
            knowledge_name: str,
            document_id: int,
            file_url: str,
            status: str,
            message: str,
            embedding_model: str,
            embedding_dim: int,
            chunk_strategy: str,
            chunk_size: int,
            token_size: int,
            chunk_count: int,
            vector_count: int,
            started_at: datetime,
            finished_at: datetime,
            stage_detail: str | None = None,
    ) -> "KnowledgeImportCallbackPayload":
        """
        功能描述:
            构造标准导入回调参数对象并自动计算处理耗时毫秒值。

        参数说明:
            task_uuid (str): 导入任务唯一标识。
            knowledge_name (str): 知识库名称。
            document_id (int): 文档 ID。
            file_url (str): 导入文件 URL。
            status (str): 回调阶段（STARTED / PROCESSING / COMPLETED / FAILED）。
            message (str): 结果摘要或失败原因。
            embedding_model (str): 向量模型名称。
            embedding_dim (int): 向量维度。
            chunk_strategy (str): 切片策略。
            chunk_size (int): 字符切片大小。
            token_size (int): token 切片大小。
            chunk_count (int): 切片总数。
            vector_count (int): 向量总数。
            started_at (datetime): 开始时间。
            finished_at (datetime): 结束时间。
            stage_detail (str | None): 可选子阶段描述。

        返回值:
            KnowledgeImportCallbackPayload: 回调参数对象。

        异常说明:
            无。字段非法时由 Pydantic 抛出验证异常。
        """
        start = started_at.astimezone(timezone.utc)
        end = finished_at.astimezone(timezone.utc)
        duration_ms = int(max(0.0, (end - start).total_seconds()) * 1000)
        return cls(
            task_uuid=task_uuid,
            knowledge_name=knowledge_name,
            document_id=document_id,
            file_url=file_url,
            status=status,
            stage_detail=stage_detail,
            message=message,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            chunk_strategy=chunk_strategy,
            chunk_size=chunk_size,
            token_size=token_size,
            chunk_count=chunk_count,
            vector_count=vector_count,
            started_at=start,
            finished_at=end,
            duration_ms=duration_ms,
        )

    def to_callback_body(self) -> dict:
        """
        功能描述:
            将回调模型序列化为 POST JSON body 字典，日期字段统一为 ISO8601 字符串。

        参数说明:
            无。

        返回值:
            dict: 适用于 HTTP POST ``json=`` 参数的字典。

        异常说明:
            无。
        """
        payload = self.model_dump(exclude_none=True)
        payload["started_at"] = self.started_at.astimezone(timezone.utc).isoformat()
        payload["finished_at"] = self.finished_at.astimezone(timezone.utc).isoformat()
        return payload
