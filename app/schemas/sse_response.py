from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class Content(BaseModel):
    text: str = Field(..., description="文本")
    node: str = Field(..., description="节点")
    message: str = Field(..., description="消息")
    name: str | None = Field(default=None, description="名称")
    arguments: str | None = Field(default=None, description="参数")


class MessageType(str, Enum):
    ANSWER = "answer"
    FUNCTION_CALL = "function_call"
    TOOL_RESPONSE = "tool_response"
    STATUS = "status"


class AssistantResponse(BaseModel):
    """AI助手SSE响应参数"""

    content: Content = Field(..., description="内容")
    type: MessageType = Field(default=MessageType.ANSWER, description="类型")
    is_end: bool = Field(default=False, description="是否结束")
    timestamp: int = Field(default_factory=lambda: int(datetime.now().timestamp()), description="时间戳")
