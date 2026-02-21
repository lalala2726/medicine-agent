from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field, field_validator


class MessageRole(str, Enum):
    """管理助手消息角色。"""

    USER = "user"
    ASSISTANT = "assistant"


class MessageStatus(str, Enum):
    """管理助手消息状态。"""

    SUCCESS = "success"
    ERROR = "error"

class ToolCallTraceItem(BaseModel):
    """单次工具调用追踪明细。"""

    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(..., min_length=1, description="工具名称")
    tool_input: Any = Field(..., description="工具输入参数")
    tool_output: Any = Field(..., description="工具输出结果")
    is_error: bool = Field(default=False, description="是否为错误调用")
    error_message: str | None = Field(default=None, description="错误信息")


class ExecutionTraceItem(BaseModel):
    """单个节点执行追踪明细。"""

    model_config = ConfigDict(extra="forbid")

    node_name: str = Field(..., min_length=1, description="节点名称")
    model_name: str = Field(..., min_length=1, description="模型名称")
    input_messages: list[Any] = Field(default_factory=list, description="节点输入消息")
    output_text: str = Field(default="", description="节点输出文本")
    tool_calls: list[ToolCallTraceItem] = Field(default_factory=list, description="节点工具调用明细")


class AdminMessageCreate(BaseModel):
    """新增管理助手消息入参模型（服务层内部使用）。"""

    model_config = ConfigDict(extra="forbid")

    uuid: str = Field(..., min_length=1, description="消息业务唯一ID（UUID）")
    conversation_id: str = Field(..., min_length=1, description="所属会话ID")
    role: MessageRole = Field(..., description="消息角色")
    status: MessageStatus = Field(default=MessageStatus.SUCCESS, description="消息状态")
    content: str = Field(..., min_length=1, description="消息内容")
    thought_chain: list[Any] | None = Field(default=None, description="思维链结构")
    execution_trace: list[ExecutionTraceItem] | None = Field(default=None, description="节点执行追踪")


class AdminMessageDocument(BaseModel):
    """MongoDB admin_messages 文档模型。"""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str | None = Field(default=None, alias="_id", description="MongoDB 主键")
    uuid: str = Field(..., description="消息业务唯一ID（UUID）")
    conversation_id: str = Field(..., description="所属会话ID")
    role: MessageRole = Field(..., description="消息角色")
    status: MessageStatus = Field(default=MessageStatus.SUCCESS, description="消息状态")
    content: str = Field(..., description="消息内容")
    thought_chain: list[Any] | None = Field(default=None, description="思维链结构")
    execution_trace: list[ExecutionTraceItem] | None = Field(default=None, description="节点执行追踪")
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
