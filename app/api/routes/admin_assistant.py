from typing import Any

from fastapi import APIRouter, Depends, Path
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from app.core.pre_authorize import RoleCode, has_permission, has_role, pre_authorize
from app.schemas.admin_assistant_history import (
    ConversationMessagesRequest,
)
from app.schemas.base_request import PageRequest
from app.schemas.response import ApiResponse, PageResponse
from app.services.admin_assisant_service import (
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


class ConversationListRequest(BaseModel):
    """智能助手会话列表请求参数。"""
    model_config = ConfigDict(extra="forbid")

    page_num: int = Field(default=1, ge=1, description="页号")
    page_size: int = Field(default=20, ge=1, le=100, description="每页大小")


class UpdateConversationTitleRequest(BaseModel):
    """修改会话标题请求参数。"""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=1, max_length=100, description="会话标题")


@router.post("/chat", summary="普通对话")
@pre_authorize(
    lambda: has_role(RoleCode.SUPER_ADMIN) or has_permission("admin:assistant:access")
)
async def assistant(request: AssistantRequest) -> StreamingResponse:
    """管理助手聊天接口（SSE 流式返回）。"""

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


@router.put("/conversation/{conversation_uuid}/title", summary="修改管理助手会话标题")
@pre_authorize(
    lambda: has_role(RoleCode.SUPER_ADMIN) or has_permission("admin:assistant:update")
)
async def update_conversation_title(
        request: UpdateConversationTitleRequest,
        conversation_uuid: str = Path(..., min_length=1, description="会话UUID"),
) -> ApiResponse[dict[str, str]]:
    """修改管理助手会话标题"""
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


@router.get("/conversation/{conversation_uuid}/messages", summary="管理助手历史消息")
@pre_authorize(
    lambda: has_role(RoleCode.SUPER_ADMIN) or has_permission("admin:assistant:access")
)
async def conversation_messages(
        conversation_uuid: str = Path(..., min_length=1, description="会话UUID"),
        request: ConversationMessagesRequest = Depends(),
) -> ApiResponse[list[dict[str, Any]]]:
    """分页查询管理助手历史消息。"""
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
