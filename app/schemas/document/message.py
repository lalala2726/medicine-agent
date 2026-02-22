from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field, field_validator


class MessageRole(str, Enum):
    """会话消息角色。"""

    USER = "user"
    AI = "ai"


class MessageStatus(str, Enum):
    """会话消息状态。"""

    SUCCESS = "success"
    ERROR = "error"


class TokenUsage(BaseModel):
    """消息级 token 使用总量。"""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = Field(default=0, ge=0, description="消息输入 Token 总数")
    completion_tokens: int = Field(default=0, ge=0, description="消息输出 Token 总数")
    total_tokens: int = Field(default=0, ge=0, description="消息总 Token 数")


class MessageCreate(BaseModel):
    """新增会话消息入参模型（服务层内部使用）。"""

    model_config = ConfigDict(extra="forbid")

    uuid: str = Field(..., min_length=1, description="消息业务唯一ID（UUID）")
    conversation_id: str = Field(..., min_length=1, description="所属会话ID")
    role: MessageRole = Field(..., description="消息角色")
    status: MessageStatus = Field(default=MessageStatus.SUCCESS, description="消息状态")
    content: str = Field(..., min_length=1, description="消息内容")
    token_usage: TokenUsage | None = Field(default=None, description="消息 token 使用总量")


class MessageDocument(BaseModel):
    """MongoDB messages 文档模型。"""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str | None = Field(default=None, alias="_id", description="MongoDB 主键")
    uuid: str = Field(..., description="消息业务唯一ID（UUID）")
    conversation_id: str = Field(..., description="所属会话ID")
    role: MessageRole = Field(..., description="消息角色")
    status: MessageStatus = Field(default=MessageStatus.SUCCESS, description="消息状态")
    content: str = Field(..., description="消息内容")
    token_usage: TokenUsage | None = Field(default=None, description="消息 token 使用总量")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")

    @field_validator("id", mode="before")
    @classmethod
    def _normalize_message_id(cls, value: Any) -> str | None:
        """把 Mongo ObjectId 统一转换为字符串。"""

        if value is None:
            return None
        if isinstance(value, ObjectId):
            return str(value)
        return str(value)

    @field_validator("conversation_id", mode="before")
    @classmethod
    def _normalize_conversation_id(cls, value: Any) -> str:
        """把 conversation_id 转为字符串，便于接口层直接透传。"""

        if isinstance(value, ObjectId):
            return str(value)
        return str(value)
