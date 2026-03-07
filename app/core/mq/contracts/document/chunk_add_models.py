from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.core.mq.contracts.document.result_stages import DocumentChunkResultStage

# 仅允许以字母开头，后续字符可包含字母、数字和下划线。
KNOWLEDGE_NAME_PATTERN = r"^[A-Za-z][A-Za-z0-9_]*$"


class KnowledgeChunkAddCommandMessage(BaseModel):
    """智能服务消费的手工新增切片命令消息模型。

    Attributes:
        message_type: 固定消息类型，值为 ``knowledge_chunk_add_command``。
        task_uuid: 业务侧生成的任务唯一标识。
        chunk_id: Admin 本地占位切片 ID，AI 必须原样回传。
        knowledge_name: Milvus collection 名称。
        document_id: 业务文档 ID。
        content: 需要新增的切片内容。
        embedding_model: 向量模型名称。
        created_at: 业务侧创建消息的时间戳。
    """

    message_type: Literal["knowledge_chunk_add_command"] = "knowledge_chunk_add_command"
    task_uuid: str = Field(..., min_length=1)
    chunk_id: int = Field(..., gt=0)
    knowledge_name: str = Field(..., pattern=KNOWLEDGE_NAME_PATTERN)
    document_id: int = Field(..., gt=0)
    content: str = Field(..., min_length=1)
    embedding_model: str = Field(..., min_length=1)
    created_at: datetime

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        """标准化并校验切片内容非空。

        Args:
            value: 原始切片内容。

        Returns:
            str: 去除首尾空白后的切片内容。

        Raises:
            ValueError: 清洗后内容为空时抛出。
        """
        normalized = value.strip()
        if not normalized:
            raise ValueError("content 不能为空")
        return normalized


class KnowledgeChunkAddResultMessage(BaseModel):
    """智能服务发布的手工新增切片结果消息模型。

    Attributes:
        message_type: 固定消息类型，值为 ``knowledge_chunk_add_result``。
        task_uuid: 对应 command 的任务 ID。
        chunk_id: 必须原样回传 Admin 下发的 chunk_id。
        stage: 当前任务阶段。
        message: 用户可读的结果描述或失败原因。
        knowledge_name: 知识库名称。
        document_id: 业务文档 ID。
        vector_id: Milvus 向量主键 ID，COMPLETED 时必须为正整数。
        chunk_index: 最终切片序号，COMPLETED 时必须为正整数。
        embedding_model: 实际执行的向量模型。
        embedding_dim: 向量维度，失败时允许为 ``0``。
        occurred_at: 结果事件产生时间。
        duration_ms: 从任务开始到当前事件的耗时。
    """

    message_type: Literal["knowledge_chunk_add_result"] = "knowledge_chunk_add_result"
    task_uuid: str = Field(..., min_length=1)
    chunk_id: int = Field(..., gt=0)
    stage: DocumentChunkResultStage
    message: str = Field(..., min_length=1)
    knowledge_name: str = Field(..., min_length=1)
    document_id: int = Field(..., gt=0)
    vector_id: int | None = Field(default=None)
    chunk_index: int | None = Field(default=None)
    embedding_model: str = Field(..., min_length=1)
    embedding_dim: int = Field(default=0, ge=0)
    occurred_at: datetime
    duration_ms: int = Field(default=0, ge=0)

    @classmethod
    def build(
            cls,
            *,
            task_uuid: str,
            chunk_id: int,
            stage: DocumentChunkResultStage,
            message: str,
            knowledge_name: str,
            document_id: int,
            embedding_model: str,
            vector_id: int | None = None,
            chunk_index: int | None = None,
            embedding_dim: int = 0,
            started_at: datetime | None = None,
            occurred_at: datetime | None = None,
    ) -> "KnowledgeChunkAddResultMessage":
        """构建标准化手工新增切片结果事件。

        Args:
            task_uuid: 任务唯一标识。
            chunk_id: Admin 本地占位切片 ID。
            stage: 当前结果阶段。
            message: 结果说明或失败原因。
            knowledge_name: 知识库名称。
            document_id: 业务文档 ID。
            embedding_model: 实际执行的向量模型名称。
            vector_id: Milvus 向量主键 ID，COMPLETED 时必填。
            chunk_index: 最终切片序号，COMPLETED 时必填。
            embedding_dim: 当前已知向量维度。
            started_at: 任务起始时间，用于计算耗时。
            occurred_at: 当前事件发生时间，未传时使用当前 UTC 时间。

        Returns:
            KnowledgeChunkAddResultMessage: 结构完整的结果消息对象。
        """
        event_time = (occurred_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
        normalized_started_at = (
            started_at.astimezone(timezone.utc) if started_at is not None else event_time
        )
        duration_ms = int(max(0.0, (event_time - normalized_started_at).total_seconds()) * 1000)
        return cls(
            task_uuid=task_uuid,
            chunk_id=chunk_id,
            stage=stage,
            message=message,
            knowledge_name=knowledge_name,
            document_id=document_id,
            vector_id=vector_id,
            chunk_index=chunk_index,
            embedding_model=embedding_model,
            embedding_dim=max(0, embedding_dim),
            occurred_at=event_time,
            duration_ms=duration_ms,
        )

    def to_json_bytes(self) -> bytes:
        """将结果消息序列化为 UTF-8 JSON 字节串。

        Returns:
            bytes: UTF-8 编码后的 JSON 消息体。
        """
        return self.model_dump_json().encode("utf-8")
