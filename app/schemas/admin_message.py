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


class TokenCounter(BaseModel):
    """统一的 token 计数结构。"""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = Field(default=0, ge=0, description="输入 Token 数")
    completion_tokens: int = Field(default=0, ge=0, description="输出 Token 数")
    total_tokens: int = Field(default=0, ge=0, description="总 Token 数")


class ToolLlmBreakdown(BaseModel):
    """工具级 LLM token 明细（支持递归 children）。"""

    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(..., min_length=1, description="工具名称")
    tool_input: Any = Field(default=None, description="工具输入参数")
    prompt_tokens: int = Field(default=0, ge=0, description="输入 Token 数")
    completion_tokens: int = Field(default=0, ge=0, description="输出 Token 数")
    total_tokens: int = Field(default=0, ge=0, description="工具总 Token 数")
    children: list["ToolLlmBreakdown"] = Field(default_factory=list, description="子工具明细")


class NodeTokenBreakdown(BaseModel):
    """节点级 token 明细。"""

    model_config = ConfigDict(extra="forbid")

    node_name: str = Field(..., min_length=1, description="节点名称")
    model_name: str = Field(..., min_length=1, description="模型名称")
    prompt_tokens: int = Field(default=0, ge=0, description="节点输入 Token 数")
    completion_tokens: int = Field(default=0, ge=0, description="节点输出 Token 数")
    total_tokens: int = Field(default=0, ge=0, description="节点总 Token 数")
    tool_tokens_total: int = Field(default=0, ge=0, description="节点下工具 LLM 总 Token 数")
    tool_llm_breakdown: list[ToolLlmBreakdown] = Field(default_factory=list, description="工具级明细")


class TokenUsage(BaseModel):
    """消息级 token 使用情况。"""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = Field(default=0, ge=0, description="消息输入 Token 总数")
    completion_tokens: int = Field(default=0, ge=0, description="消息输出 Token 总数")
    total_tokens: int = Field(default=0, ge=0, description="消息总 Token 数")
    is_complete: bool = Field(default=True, description="本轮所有 LLM 调用是否都有 usage")
    node_breakdown: list[NodeTokenBreakdown] = Field(default_factory=list, description="节点级明细")


class ToolCallTraceItem(BaseModel):
    """单次工具调用追踪明细。"""

    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(..., min_length=1, description="工具名称")
    tool_input: Any = Field(default=None, description="工具输入参数")
    is_error: bool = Field(default=False, description="是否为错误调用")
    error_message: str | None = Field(default=None, description="错误信息")
    llm_used: bool = Field(default=False, description="该工具内部是否触发 LLM 调用")
    llm_usage_complete: bool = Field(default=True, description="该工具内部 LLM usage 是否完整")
    llm_token_usage: TokenCounter | None = Field(default=None, description="工具内部 LLM token")
    children: list["ToolCallTraceItem"] = Field(default_factory=list, description="子工具调用")


class ExecutionTraceItem(BaseModel):
    """单个节点执行追踪明细。"""

    model_config = ConfigDict(extra="forbid")

    node_name: str = Field(..., min_length=1, description="节点名称")
    model_name: str = Field(..., min_length=1, description="模型名称")
    input_messages: list[Any] = Field(default_factory=list, description="节点输入消息")
    output_text: str = Field(default="", description="节点输出文本")
    llm_used: bool = Field(default=True, description="节点是否触发 LLM 调用")
    llm_usage_complete: bool = Field(default=True, description="节点 LLM usage 是否完整")
    llm_token_usage: TokenCounter | None = Field(default=None, description="节点自身 LLM token")
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
    token_usage: TokenUsage | None = Field(default=None, description="消息 token 使用明细")
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
    token_usage: TokenUsage | None = Field(default=None, description="消息 token 使用明细")
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


ToolCallTraceItem.model_rebuild()
ToolLlmBreakdown.model_rebuild()
