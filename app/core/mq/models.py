from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field

from app.rag.chunking import ChunkStrategyType


class KnowledgeImportMessage(BaseModel):
    """
    功能描述:
        定义知识库导入任务的 MQ 消息结构。

    参数说明:
        task_uuid (str): 任务唯一标识。
        knowledge_name (str): 知识库名称。
        document_id (int): 文档 ID。
        file_url (str): 单个文件 URL。
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
