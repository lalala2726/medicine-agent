from __future__ import annotations

from fastapi import APIRouter, WebSocket

from app.core.security.auth_context import (
    reset_authorization_header,
    reset_current_user,
    set_authorization_header,
    set_current_user,
)
from app.core.security.pre_authorize import RoleCode
from app.schemas.auth import AuthUser
from app.services.speech_stt_service import speech_stt_stream_service
from app.services.auth_service import verify_authorization

router = APIRouter(prefix="/ws/speech/stt", tags=["语音识别"])

SPEECH_STT_ACCESS_PERMISSION = "admin:assistant:access"


def _has_stt_access(user: AuthUser) -> bool:
    if RoleCode.SUPER_ADMIN.value in user.roles:
        return True
    return SPEECH_STT_ACCESS_PERMISSION in user.permissions


def _resolve_query_authorization(websocket: WebSocket) -> str | None:
    raw_token = (
        websocket.query_params.get("access_token")
        or websocket.query_params.get("token")
        or ""
    ).strip()
    if not raw_token:
        return None
    if raw_token.lower().startswith("bearer "):
        return raw_token
    return f"Bearer {raw_token}"


@router.websocket("/stream")
async def speech_stt_stream(websocket: WebSocket) -> None:
    """
    语音识别 STT WebSocket 接口。

    认证规则：
    - 从 query 参数读取 token（`access_token` 或 `token`）；
    - 鉴权成功后转发到 STT 服务。
    """

    auth_token = set_authorization_header(_resolve_query_authorization(websocket))
    user_token = set_current_user(None)
    try:
        try:
            current_user = await verify_authorization()
        except Exception:
            await websocket.close(code=1008, reason="unauthorized")
            return

        reset_current_user(user_token)
        user_token = set_current_user(current_user)
        if not _has_stt_access(current_user):
            await websocket.close(code=1008, reason="forbidden")
            return

        await speech_stt_stream_service(
            websocket=websocket,
            user=current_user,
        )
    finally:
        reset_current_user(user_token)
        reset_authorization_header(auth_token)


__all__ = ["router"]
