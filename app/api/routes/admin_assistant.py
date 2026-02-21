from typing import Any

from fastapi import APIRouter, Depends, Path
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.pre_authorize import RoleCode, has_permission, has_role, pre_authorize
from app.schemas.admin_assistant_history import (
    ConversationMessagesRequest,
)
from app.schemas.base_request import PageRequest
from app.schemas.response import ApiResponse, PageResponse
from app.services.admin_assistant_service import (
    assistant_chat,
    conversation_list as conversation_list_service,
    conversation_messages as conversation_messages_service,
    delete_conversation as delete_conversation_service,
    update_conversation_title as update_conversation_title_service,
)

router = APIRouter(prefix="/admin/assistant", tags=["智能助手"])


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


class ConversationListRequest(BaseModel):
    """智能助手会话列表请求参数。"""
    model_config = ConfigDict(extra="forbid")

    page_num: int = Field(default=1, ge=1, description="页号")
    page_size: int = Field(default=20, ge=1, le=100, description="每页大小")


class UpdateConversationTitleRequest(BaseModel):
    """修改会话标题请求参数。"""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=1, max_length=100, description="会话标题")


def _build_conversation_messages_response(
        *,
        conversation_uuid: str,
        request: ConversationMessagesRequest,
) -> ApiResponse[list[dict[str, Any]]]:
    """
    统一构造会话消息分页响应，供新旧路由复用。

    Args:
        conversation_uuid: 会话 UUID。
        request: 分页请求参数（`page_num/page_size`）。

    Returns:
        ApiResponse[list[dict[str, Any]]]: 序列化后的历史消息数组响应。
    """

    messages = conversation_messages_service(
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
    return ApiResponse.success(data=serialized_messages)


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
@pre_authorize(
    lambda: has_role(RoleCode.SUPER_ADMIN) or has_permission("admin:assistant:access")
)
async def assistant(request: AssistantRequest) -> StreamingResponse:
    """管理助手聊天入口（SSE 流式返回，基于 Gateway + Supervisor 工作流）。"""

    return assistant_chat(
        question=request.question,
        conversation_uuid=request.conversation_uuid,
    )


@router.get("/conversation/list", summary="管理助手会话列表")
@pre_authorize(
    lambda: has_role(RoleCode.SUPER_ADMIN) or has_permission("admin:assistant:access")
)
async def conversation_list(
        request: ConversationListRequest = Depends(),
) -> ApiResponse[PageResponse[dict[str, str]]]:
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
    修改管理助手会话标题（新路径）。

    说明：保留该路径以兼容当前实现，同时在 `/conversation/{conversation_uuid}/title`
    提供等价旧路径，避免历史客户端与测试用例出现 404。
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
) -> ApiResponse[list[dict[str, Any]]]:
    """
    分页查询管理助手历史消息（新路径）。

    说明：保留该路径以兼容当前实现，同时在 `/conversation/{conversation_uuid}/messages`
    提供等价旧路径，避免历史客户端与测试用例出现 404。
    """

    return _build_conversation_messages_response(
        conversation_uuid=conversation_uuid,
        request=request,
    )


@router.put("/conversation/{conversation_uuid}/title", summary="修改管理助手会话标题（兼容路径）")
@pre_authorize(
    lambda: has_role(RoleCode.SUPER_ADMIN) or has_permission("admin:assistant:update")
)
async def update_conversation_title_legacy(
        request: UpdateConversationTitleRequest,
        conversation_uuid: str = Path(..., min_length=1, description="会话UUID"),
) -> ApiResponse[dict[str, str]]:
    """
    修改会话标题（旧路径兼容）。

    Args:
        request: 标题更新请求对象。
        conversation_uuid: 会话 UUID。

    Returns:
        ApiResponse[dict[str, str]]: 修改结果，结构与新路径一致。
    """

    return _build_update_conversation_title_response(
        conversation_uuid=conversation_uuid,
        request=request,
    )


@router.get("/conversation/{conversation_uuid}/messages", summary="管理助手历史消息（兼容路径）")
@pre_authorize(
    lambda: has_role(RoleCode.SUPER_ADMIN) or has_permission("admin:assistant:access")
)
async def conversation_messages_legacy(
        conversation_uuid: str = Path(..., min_length=1, description="会话UUID"),
        request: ConversationMessagesRequest = Depends(),
) -> ApiResponse[list[dict[str, Any]]]:
    """
    查询会话历史消息（旧路径兼容）。

    Args:
        conversation_uuid: 会话 UUID。
        request: 分页请求参数。

    Returns:
        ApiResponse[list[dict[str, Any]]]: 会话历史消息列表响应。
    """

    return _build_conversation_messages_response(
        conversation_uuid=conversation_uuid,
        request=request,
    )
