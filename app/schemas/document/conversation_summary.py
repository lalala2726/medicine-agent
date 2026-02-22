from __future__ import annotations

from datetime import datetime
from typing import Any

from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ConversationSummary(BaseModel):
    """MongoDB conversation_summaries 文档模型。"""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str | None = Field(default=None, alias="_id", description="MongoDB 主键")
    conversation_id: str = Field(..., description="所属会话ID")
    summary_content: str = Field(..., description="总结内容")
    summarized_messages: list[str] = Field(default_factory=list, description="该总结包含的消息 UUID 列表")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    status: str = Field(..., description="总结状态（如已总结、失败等）")

    @field_validator("id", mode="before")
    @classmethod
    def _normalize_id(cls, value: Any) -> str | None:
        """把 Mongo ObjectId 统一转换为字符串。"""

        if value is None:
            return None
        if isinstance(value, ObjectId):
            return str(value)
        return str(value)

    @field_validator("conversation_id", mode="before")
    @classmethod
    def _normalize_conversation_id(cls, value: Any) -> str:
        """把 conversation_id 转为字符串。"""

        if isinstance(value, ObjectId):
            return str(value)
        return str(value)
