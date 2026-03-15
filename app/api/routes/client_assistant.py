from typing import Any

from fastapi import APIRouter, Depends, Path, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.security.rate_limit import RateLimitPreset, RateLimitRule, rate_limit
from app.schemas.admin_assistant_history import ConversationMessagesRequest
from app.schemas.base_request import PageRequest
from app.schemas.document.conversation import ConversationListItem
from app.schemas.response import ApiResponse, PageResponse
from app.services.client_assistant_service import (
    assistant_chat,
    conversation_list as conversation_list_service,
    conversation_messages as conversation_messages_service,
)

router = APIRouter(prefix="/client/assistant", tags=["客户端助手"])

CHAT_RATE_LIMIT_RULES = (
    RateLimitRule.preset(RateLimitPreset.MINUTE_1, limit=10),
    RateLimitRule.preset(RateLimitPreset.MINUTE_5, limit=30),
    RateLimitRule.preset(RateLimitPreset.HOUR_1, limit=120),
    RateLimitRule.preset(RateLimitPreset.HOUR_24, limit=600),
)


class ClientAssistantRequest(BaseModel):
    """客户端助手请求参数。"""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "question": "我想申请退款，现在应该怎么操作？",
                "conversation_uuid": "b7e4f95d-62f6-4a0a-8823-2fbd3d4db0cf",
            }
        },
    )

    question: str = Field(..., min_length=1, description="问题")
    conversation_uuid: str | None = Field(
        default=None,
        min_length=1,
        description="会话UUID",
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("问题不能为空")
        return normalized

    @field_validator("conversation_uuid")
    @classmethod
    def validate_conversation_uuid(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class ConversationListRequest(BaseModel):
    """客户端助手会话列表请求参数。"""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "page_num": 1,
                "page_size": 20,
            }
        },
    )

    page_num: int = Field(default=1, ge=1, description="页号")
    page_size: int = Field(default=20, ge=1, le=100, description="每页大小")


@router.post(
    "/chat",
    summary="客户端助手对话",
    description=(
            "客户端 AI 助手聊天接口，使用 `POST + text/event-stream` 返回 SSE 流。"
            "新会话会先推送带 `conversation_uuid` 和 `message_uuid` 的 `notice` 事件；"
            "已存在会话会先推送带 `message_uuid` 的 `notice` 事件；"
            "随后连续推送 `answer/thinking` 增量事件，最终以 `is_end=true` 的结束事件收尾。"
    ),
    response_description="SSE 流式事件；前端需按 `data: <json>\\n\\n` 解析。",
)
@rate_limit(
    rules=CHAT_RATE_LIMIT_RULES,
    subjects=("user_id",),
    scope="client_assistant_chat",
    fail_open=False,
)
async def assistant(_request: Request, request: ClientAssistantRequest) -> StreamingResponse:
    """
    客户端助手聊天入口（SSE 流式返回）。

    对接要点：
    1. 这是 `POST` 接口，标准浏览器 `EventSource` 不能直接使用；
    2. 请使用 `fetch`/`ReadableStream` 按 SSE 协议解析；
    3. 当前版本主要事件类型为 `notice`、`answer`、`thinking`；
    4. 结束标志为最后一条 `is_end=true` 的事件。
    """

    return assistant_chat(
        question=request.question,
        conversation_uuid=request.conversation_uuid,
    )


@router.get(
    "/conversation/list",
    summary="客户端助手会话列表",
    description="分页查询当前登录用户的客户端助手会话列表，仅返回 `conversation_uuid` 和 `title`。",
)
async def conversation_list(
        request: ConversationListRequest = Depends(),
) -> ApiResponse[PageResponse[ConversationListItem]]:
    """分页查询客户端助手会话列表。"""

    rows, total = conversation_list_service(
        page_request=PageRequest(
            page_num=request.page_num,
            page_size=request.page_size,
        )
    )
    return ApiResponse.page(
        rows=rows,
        total=total,
        page_num=request.page_num,
        page_size=request.page_size,
    )


@router.get(
    "/history/{conversation_uuid}",
    summary="客户端助手历史消息",
    description=(
            "分页查询指定客户端助手会话的历史消息。"
            "返回结果按时间正序排列，可直接用于前端会话回放。"
    ),
)
async def conversation_messages(
        conversation_uuid: str = Path(..., min_length=1, description="会话UUID"),
        request: ConversationMessagesRequest = Depends(),
) -> ApiResponse[PageResponse[dict[str, Any]]]:
    """分页查询客户端助手历史消息。"""

    messages, total = conversation_messages_service(
        conversation_uuid=conversation_uuid,
        page_request=PageRequest(
            page_num=request.page_num,
            page_size=request.page_size,
        ),
    )
    serialized_messages = [
        message.model_dump(by_alias=True, exclude_none=True)
        for message in messages
    ]
    return ApiResponse.page(
        rows=serialized_messages,
        total=total,
        page_num=request.page_num,
        page_size=request.page_size,
    )
