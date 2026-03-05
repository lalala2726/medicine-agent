from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.rag.chunking import ChunkStrategyType

DEFAULT_COMMAND_CHUNK_SIZE = 500  # command 默认字符切片大小
DEFAULT_COMMAND_TOKEN_SIZE = 100  # command 默认 token 切片大小


class ImportResultStage(str, Enum):
    """导入结果事件阶段枚举。"""

    STARTED = "STARTED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ProcessingStageDetail(str, Enum):
    """`PROCESSING` 阶段的细分步骤枚举。"""

    DOWNLOADING = "downloading"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INSERTING = "inserting"


class KnowledgeImportCommandMessage(BaseModel):
    """智能服务消费的导入命令消息模型。"""

    message_type: Literal["knowledge_import_command"] = "knowledge_import_command"
    task_uuid: str = Field(..., min_length=1)
    biz_key: str = Field(..., min_length=1)
    version: int = Field(..., ge=1)
    knowledge_name: str = Field(..., min_length=1)
    document_id: int = Field(..., gt=0)
    file_url: str = Field(..., min_length=1)
    embedding_model: str = Field(..., min_length=1)
    chunk_strategy: ChunkStrategyType = Field(default=ChunkStrategyType.CHARACTER)
    chunk_size: int = Field(default=DEFAULT_COMMAND_CHUNK_SIZE, ge=1)
    token_size: int = Field(default=DEFAULT_COMMAND_TOKEN_SIZE, ge=1)
    created_at: datetime

    @model_validator(mode="before")
    @classmethod
    def _normalize_nullable_chunk_options(cls, data: Any) -> Any:
        """标准化可空切片参数，避免无关字段导致消息校验失败。

        Args:
            data: 原始入参字典。

        Returns:
            Any: 标准化后的入参。
        """
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        if payload.get("chunk_size") is None:
            payload["chunk_size"] = DEFAULT_COMMAND_CHUNK_SIZE
        if payload.get("token_size") is None:
            payload["token_size"] = DEFAULT_COMMAND_TOKEN_SIZE
        return payload

    def to_json_bytes(self) -> bytes:
        """将命令消息序列化为 UTF-8 JSON 字节串。

        Returns:
            bytes: 序列化后的消息体。
        """
        return self.model_dump_json().encode("utf-8")


class KnowledgeImportResultMessage(BaseModel):
    """智能服务发布的导入结果消息模型。"""

    message_type: Literal["knowledge_import_result"] = "knowledge_import_result"
    task_uuid: str = Field(..., min_length=1)
    biz_key: str = Field(..., min_length=1)
    version: int = Field(..., ge=1)
    stage: ImportResultStage
    stage_detail: ProcessingStageDetail | None = None
    message: str = Field(..., min_length=1)
    knowledge_name: str = Field(..., min_length=1)
    document_id: int = Field(..., gt=0)
    file_url: str = Field(..., min_length=1)
    filename: str | None = None
    chunk_count: int = Field(default=0, ge=0)
    vector_count: int = Field(default=0, ge=0)
    embedding_model: str = Field(..., min_length=1)
    embedding_dim: int = Field(default=0, ge=0)
    occurred_at: datetime
    duration_ms: int = Field(default=0, ge=0)

    @classmethod
    def build(
            cls,
            *,
            task_uuid: str,
            biz_key: str,
            version: int,
            stage: ImportResultStage,
            message: str,
            knowledge_name: str,
            document_id: int,
            file_url: str,
            embedding_model: str,
            stage_detail: ProcessingStageDetail | None = None,
            filename: str | None = None,
            chunk_count: int = 0,
            vector_count: int = 0,
            embedding_dim: int = 0,
            started_at: datetime | None = None,
            occurred_at: datetime | None = None,
    ) -> "KnowledgeImportResultMessage":
        """构建标准化导入结果事件。

        Args:
            task_uuid: 任务唯一标识。
            biz_key: 业务对象唯一键。
            version: 同一 biz_key 下的递增版本号。
            stage: 当前事件阶段。
            message: 阶段消息描述。
            knowledge_name: 知识库名称。
            document_id: 文档 ID。
            file_url: 文件 URL。
            embedding_model: 向量模型名称。
            stage_detail: 可选处理子阶段。
            filename: 可选下载文件名。
            chunk_count: 成功时切片总数。
            vector_count: 成功时向量总数。
            embedding_dim: 实际向量维度。
            started_at: 可选任务开始时间。
            occurred_at: 可选事件发生时间。

        Returns:
            KnowledgeImportResultMessage: 构建后的结果事件对象。
        """
        event_time = (occurred_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
        normalized_started_at = (
            started_at.astimezone(timezone.utc) if started_at is not None else event_time
        )
        duration_ms = int(max(0.0, (event_time - normalized_started_at).total_seconds()) * 1000)
        return cls(
            task_uuid=task_uuid,
            biz_key=biz_key,
            version=version,
            stage=stage,
            stage_detail=stage_detail,
            message=message,
            knowledge_name=knowledge_name,
            document_id=document_id,
            file_url=file_url,
            filename=filename,
            chunk_count=chunk_count,
            vector_count=vector_count,
            embedding_model=embedding_model,
            embedding_dim=max(0, embedding_dim),
            occurred_at=event_time,
            duration_ms=duration_ms,
        )

    def to_json_bytes(self) -> bytes:
        """将结果消息序列化为 UTF-8 JSON 字节串。

        Returns:
            bytes: 序列化后的消息体。
        """
        return self.model_dump_json().encode("utf-8")
