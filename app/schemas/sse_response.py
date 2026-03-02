import time
from enum import Enum
from typing import Any

from pydantic import AliasChoices, BaseModel, Field


class Content(BaseModel):
    text: str | None = Field(default=None, description="文本")
    node: str | None = Field(default=None, description="节点")
    parent_node: str | None = Field(default=None, description="所属节点")
    state: str | None = Field(default=None, description="状态")
    message: str | None = Field(default=None, description="消息")
    result: str | None = Field(default=None, description="结果")
    name: str | None = Field(default=None, description="名称")
    arguments: str | None = Field(default=None, description="参数")


class MessageType(str, Enum):
    ANSWER = "answer"
    THINKING = "thinking"
    FUNCTION_CALL = "function_call"
    TOOL_RESPONSE = "tool_response"
    STATUS = "status"
    NOTICE = "notice"


class AssistantResponse(BaseModel):
    """AI助手SSE响应参数"""

    content: Content = Field(default_factory=Content, description="内容")
    type: MessageType = Field(default=MessageType.ANSWER, description="类型")
    meta: dict[str, Any] | None = Field(
        default=None,
        validation_alias=AliasChoices("meta", "extra"),
        description="元数据（如会话ID、追踪ID等扩展信息）",
    )
    is_end: bool = Field(default=False, description="是否结束")
    timestamp: int = Field(
        default_factory=lambda: int(time.time() * 1000), description="时间戳（毫秒）"
    )
