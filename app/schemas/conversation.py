from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ConversationType(str, Enum):
    """会话类型。"""

    CLIENT = "client"
    ADMIN = "admin"


class ConversationCreate(BaseModel):
    """新增会话入参模型（服务层内部使用）。"""

    model_config = ConfigDict(extra="forbid")

    uuid: str = Field(..., min_length=1, description="会话业务唯一ID")
    conversation_type: ConversationType = Field(..., description="会话类型")
    user_id: int = Field(..., ge=1, description="用户ID（int64）")
    title: str = Field(default="新聊天", description="会话标题")
    message_count: int = Field(default=0, ge=0, description="消息总数")
    metadata: dict[str, Any] | None = Field(default=None, description="扩展信息")


class ConversationDocument(BaseModel):
    """MongoDB conversations 文档模型。"""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str | None = Field(default=None, alias="_id", description="MongoDB 主键")
    uuid: str = Field(..., description="会话业务唯一ID")
    conversation_type: ConversationType = Field(..., description="会话类型")
    user_id: int = Field(..., description="用户ID（int64）")
    title: str | None = Field(default=None, description="会话标题")
    create_time: datetime = Field(..., description="创建时间")
    update_time: datetime = Field(..., description="更新时间")
    message_count: int = Field(default=0, ge=0, description="消息总数")
    metadata: dict[str, Any] | None = Field(default=None, description="扩展信息")

    @field_validator("id", mode="before")
    @classmethod
    def _normalize_object_id(cls, value: Any) -> str | None:
        """把 Mongo ObjectId 统一转换为字符串，避免上层重复处理。"""

        if value is None:
            return None
        if isinstance(value, ObjectId):
            return str(value)
        return str(value)
