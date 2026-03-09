from typing import Any, Literal

from fastapi import APIRouter, Depends, Path, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.security.anonymous_access import allow_anonymous
from app.core.security.pre_authorize import RoleCode, has_permission, has_role, pre_authorize
from app.core.security.rate_limit import RateLimitPreset, RateLimitRule, rate_limit
from app.schemas.admin_assistant_history import (
    ConversationMessagesRequest,
)
from app.schemas.base_request import PageRequest
from app.schemas.document.conversation import ConversationListItem
from app.schemas.response import ApiResponse, PageResponse
from app.services.admin_assistant_service import (
    assistant_message_tts_stream as assistant_message_tts_stream_service,
    assistant_chat,
    conversation_list as conversation_list_service,
    conversation_messages as conversation_messages_service,
    delete_conversation as delete_conversation_service,
    update_conversation_title as update_conversation_title_service,
)

router = APIRouter(prefix="/admin/assistant", tags=["智能助手"])

# 聊天率限制规则
CHAT_RATE_LIMIT_RULES = (
    RateLimitRule.preset(RateLimitPreset.MINUTE_1, limit=10),
    RateLimitRule.preset(RateLimitPreset.MINUTE_5, limit=30),
    RateLimitRule.preset(RateLimitPreset.HOUR_1, limit=120),
    RateLimitRule.preset(RateLimitPreset.HOUR_24, limit=600),
)

# 语音合成率限制规则
TTS_RATE_LIMIT_RULES = (
    RateLimitRule.preset(RateLimitPreset.MINUTE_1, limit=5),
    RateLimitRule.preset(RateLimitPreset.HOUR_1, limit=60),
    RateLimitRule.preset(RateLimitPreset.HOUR_5, limit=100),
    RateLimitRule.preset(RateLimitPreset.HOUR_24, limit=200),
)

# 测试率限制规则
TEST_RATE_LIMIT_RULES = (
    RateLimitRule.preset(RateLimitPreset.MINUTE_1, limit=10),
)

# 按照用户 ID 进行限流的主体
USER_ID_RATE_LIMIT_SUBJECTS: tuple[Literal["user_id"], ...] = ("user_id",)

# 按照 IP 进行限流的主体
IP_RATE_LIMIT_SUBJECTS: tuple[Literal["ip"], ...] = ("ip",)


class AssistantRequest(BaseModel):
    """AI助手请求参数。"""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1, description="问题")
    conversation_uuid: str | None = Field(
        default=None,
        min_length=1,
        description="会话UUID",
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """
        标准化用户问题文本。

        作用：
        1. 去掉首尾空白，避免把仅空格内容传入聊天主流程；
        2. 在路由层提前阻断无效请求，减少下游 service 分支处理复杂度。
        """

        normalized = value.strip()
        if not normalized:
            raise ValueError("问题不能为空")
        return normalized

    @field_validator("conversation_uuid")
    @classmethod
    def validate_conversation_uuid(cls, value: str | None) -> str | None:
        """
        标准化会话 UUID。

        作用：
        1. 去掉首尾空白，确保 service 层拿到的 UUID 稳定；
        2. 对于纯空白 UUID，统一视为 None，表示创建新会话。
        """

        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class AssistantMessageTtsRequest(BaseModel):
    """管理助手消息转语音请求参数。"""

    model_config = ConfigDict(extra="forbid")

    message_uuid: str = Field(..., min_length=1, description="消息 UUID")

    @field_validator("message_uuid")
    @classmethod
    def validate_message_uuid(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("message_uuid 不能为空")
        return normalized


class ConversationListRequest(BaseModel):
    """智能助手会话列表请求参数。"""
    model_config = ConfigDict(extra="forbid")

    page_num: int = Field(default=1, ge=1, description="页号")
    page_size: int = Field(default=20, ge=1, le=100, description="每页大小")


class UpdateConversationTitleRequest(BaseModel):
    """修改会话标题请求参数。"""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=1, max_length=100, description="会话标题")


def _build_update_conversation_title_response(
        *,
        conversation_uuid: str,
        request: UpdateConversationTitleRequest,
) -> ApiResponse[dict[str, str]]:
    """
    统一构造会话标题修改响应，供新旧路由复用。

    Args:
        conversation_uuid: 会话 UUID。
        request: 标题更新请求对象，读取 `title` 字段。

    Returns:
        ApiResponse[dict[str, str]]: 修改结果，包含会话 UUID 与标准化标题。
    """

    normalized_title = update_conversation_title_service(
        conversation_uuid=conversation_uuid,
        title=request.title,
    )
    return ApiResponse.success(
        data={
            "conversation_uuid": conversation_uuid,
            "title": normalized_title,
        },
        message="修改成功",
    )


@router.post("/chat", summary="管理助手对话（Gateway + Supervisor）")
@rate_limit(
    rules=CHAT_RATE_LIMIT_RULES,
    subjects=USER_ID_RATE_LIMIT_SUBJECTS,
    scope="admin_assistant_chat",
    fail_open=False,
)
@pre_authorize(
    lambda: has_role(RoleCode.SUPER_ADMIN) or has_permission("admin:assistant:access")
)
async def assistant(_request: Request, request: AssistantRequest) -> StreamingResponse:
    """
    管理助手聊天入口（SSE 流式返回，基于 Gateway + Supervisor 工作流）。

    Args:
        _request: FastAPI 原始请求对象（当前实现仅用于依赖注入与中间件链路）。
        request: 聊天请求体，包含问题与可选会话 UUID。

    Returns:
        StreamingResponse: SSE 流式响应对象。
    """

    return assistant_chat(
        question=request.question,
        conversation_uuid=request.conversation_uuid,
    )


@router.post("/message/tts/stream", summary="管理助手消息转语音（流式）")
@rate_limit(
    rules=TTS_RATE_LIMIT_RULES,
    subjects=USER_ID_RATE_LIMIT_SUBJECTS,
    scope="admin_assistant_tts",
    fail_open=False,
)
@pre_authorize(
    lambda: has_role(RoleCode.SUPER_ADMIN) or has_permission("admin:assistant:access")
)
async def assistant_message_tts_stream(
        _request: Request,
        request: AssistantMessageTtsRequest,
) -> StreamingResponse:
    """根据消息 UUID 生成语音并以 HTTP chunked 流式返回音频数据。"""

    return assistant_message_tts_stream_service(
        message_uuid=request.message_uuid,
    )


@router.get("/rate-limit/test", summary="管理助手限流测试接口")
@allow_anonymous
@rate_limit(
    rules=TEST_RATE_LIMIT_RULES,
    subjects=IP_RATE_LIMIT_SUBJECTS,
    scope="admin_assistant_rate_limit_test",
    fail_open=False,
)
async def assistant_rate_limit_test(request: Request, response: Response) -> ApiResponse[dict[str, str]]:
    """限流测试接口：用于验证每分钟 10 次限流配置。"""

    return ApiResponse.success(
        data={"status": "ok"},
        message="限流测试通过",
    )


@router.get("/conversation/list", summary="管理助手会话列表")
@pre_authorize(
    lambda: has_role(RoleCode.SUPER_ADMIN) or has_permission("admin:assistant:access")
)
async def conversation_list(
        request: ConversationListRequest = Depends(),
) -> ApiResponse[PageResponse[ConversationListItem]]:
    """
    分页查询管理助手会话列表（仅返回会话 UUID 与标题）。
    """

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


@router.delete("/conversation/{conversation_uuid}", summary="删除管理助手会话")
@pre_authorize(
    lambda: has_role(RoleCode.SUPER_ADMIN) or has_permission("admin:assistant:delete")
)
async def delete_conversation(
        conversation_uuid: str = Path(..., min_length=1, description="会话UUID"),
) -> ApiResponse[dict[str, str]]:
    """删除管理助手会话"""
    delete_conversation_service(conversation_uuid=conversation_uuid)
    return ApiResponse.success(
        data={"conversation_uuid": conversation_uuid},
        message="删除成功",
    )


@router.put("/conversation/{conversation_uuid}", summary="修改管理助手会话标题")
@pre_authorize(
    lambda: has_role(RoleCode.SUPER_ADMIN) or has_permission("admin:assistant:update")
)
async def update_conversation_title(
        request: UpdateConversationTitleRequest,
        conversation_uuid: str = Path(..., min_length=1, description="会话UUID"),
) -> ApiResponse[dict[str, str]]:
    """
    修改管理助手会话标题。
    """

    return _build_update_conversation_title_response(
        conversation_uuid=conversation_uuid,
        request=request,
    )


@router.get("/history/{conversation_uuid}", summary="管理助手历史消息")
@pre_authorize(
    lambda: has_role(RoleCode.SUPER_ADMIN) or has_permission("admin:assistant:access")
)
async def conversation_messages(
        conversation_uuid: str = Path(..., min_length=1, description="会话UUID"),
        request: ConversationMessagesRequest = Depends(),
) -> ApiResponse[PageResponse[dict[str, Any]]]:
    """
    分页查询管理助手历史消息。
    """

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

