from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field, field_validator


class MessageTraceProvider(str, Enum):
    """消息追踪使用的模型厂家。"""

    OPENAI = "openai"
    ALIYUN = "aliyun"
    VOLCENGINE = "volcengine"


class TokenCounter(BaseModel):
    """统一的 token 计数结构。"""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = Field(default=0, ge=0, description="输入 Token 数")
    completion_tokens: int = Field(default=0, ge=0, description="输出 Token 数")
    total_tokens: int = Field(default=0, ge=0, description="总 Token 数")


class TraceTokenUsage(BaseModel):
    """消息级 token 汇总结构。"""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = Field(default=0, ge=0, description="消息输入 Token 总数")
    completion_tokens: int = Field(default=0, ge=0, description="消息输出 Token 总数")
    total_tokens: int = Field(default=0, ge=0, description="消息总 Token 数")
    is_complete: bool = Field(default=True, description="本轮 LLM usage 是否完整")


class ToolCallTraceItem(BaseModel):
    """单次工具调用追踪明细。"""

    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(..., min_length=1, description="工具名称")
    tool_call_id: str | None = Field(default=None, description="工具调用 ID")
    tool_input: Any = Field(default=None, description="工具输入参数")


class ExecutionTraceItem(BaseModel):
    """单个节点执行追踪明细。"""

    model_config = ConfigDict(extra="forbid")

    sequence: int = Field(..., ge=1, description="节点执行顺序（从 1 开始）")
    node_name: str = Field(..., min_length=1, description="节点名称")
    model_name: str = Field(..., min_length=1, description="模型名称")
    status: Literal["success", "error", "cancelled"] = Field(
        default="success",
        description="节点执行状态",
    )
    output_text: str = Field(default="", description="节点输出文本")
    llm_usage_complete: bool = Field(default=True, description="节点 LLM usage 是否完整")
    llm_token_usage: TokenCounter | None = Field(default=None, description="节点自身 LLM token")
    tool_calls: list[ToolCallTraceItem] = Field(default_factory=list, description="节点工具调用明细")
    node_context: dict[str, Any] | None = Field(default=None, description="节点扩展上下文")


class WorkflowTraceSummary(BaseModel):
    """工作流级追踪汇总。"""

    model_config = ConfigDict(extra="forbid")

    workflow_name: str = Field(default="admin_assistant_graph", min_length=1, description="工作流名称")
    workflow_status: Literal["success", "error", "cancelled", "waiting_input"] = Field(
        default="success",
        description="工作流执行状态",
    )
    execution_path: list[str] = Field(default_factory=list, description="节点执行路径")
    final_node: str | None = Field(default=None, description="最终执行节点")
    route_targets: list[str] = Field(default_factory=list, description="gateway 路由目标")
    task_difficulty: Literal["normal", "high"] | None = Field(default=None, description="任务难度")


class MessageTraceCreate(BaseModel):
    """新增 message trace 入参模型（服务层内部使用）。"""

    model_config = ConfigDict(extra="forbid")

    message_uuid: str = Field(..., min_length=1, description="消息 UUID")
    conversation_id: str = Field(..., min_length=1, description="所属会话 ID")
    provider: MessageTraceProvider = Field(..., description="模型厂家")
    workflow: WorkflowTraceSummary = Field(..., description="工作流追踪汇总")
    execution_trace: list[ExecutionTraceItem] = Field(default_factory=list, description="节点执行追踪")
    token_usage: TraceTokenUsage | None = Field(default=None, description="消息级 token 汇总")


class MessageTraceDocument(BaseModel):
    """MongoDB message_traces 文档模型。"""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str | None = Field(default=None, alias="_id", description="MongoDB 主键")
    message_uuid: str = Field(..., description="消息 UUID")
    conversation_id: str = Field(..., description="所属会话 ID")
    provider: MessageTraceProvider = Field(..., description="模型厂家")
    workflow: WorkflowTraceSummary = Field(..., description="工作流追踪汇总")
    execution_trace: list[ExecutionTraceItem] = Field(default_factory=list, description="节点执行追踪")
    token_usage: TraceTokenUsage | None = Field(default=None, description="消息级 token 汇总")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")

    @field_validator("id", mode="before")
    @classmethod
    def _normalize_id(cls, value: Any) -> str | None:
        """
        功能描述：
            统一标准化 Mongo 文档主键 `_id`，确保对外字段始终为字符串类型。

        参数说明：
            value (Any): 原始 `_id` 值，可能为 `ObjectId`、字符串或其他可转字符串对象。

        返回值：
            str | None:
                - 传入为空时返回 `None`；
                - 其余场景返回字符串形式的主键值。

        异常说明：
            无。
        """

        if value is None:
            return None
        if isinstance(value, ObjectId):
            return str(value)
        return str(value)

    @field_validator("conversation_id", mode="before")
    @classmethod
    def _normalize_conversation_id(cls, value: Any) -> str:
        """
        功能描述：
            将 `conversation_id` 统一转换为字符串，避免接口层和持久层类型不一致。

        参数说明：
            value (Any): 原始会话 ID，可能为 `ObjectId` 或字符串。

        返回值：
            str: 规范化后的会话 ID 字符串。

        异常说明：
            无。
        """

        if isinstance(value, ObjectId):
            return str(value)
        return str(value)
