from pydantic import BaseModel, Field


class AssistantResponse(BaseModel):
    """AI助手SSE响应参数"""

    content: str = Field(..., description="答案")
    is_end: bool = Field(default=False, description="是否结束")
