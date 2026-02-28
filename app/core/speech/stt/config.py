from __future__ import annotations

import os
import uuid
from dataclasses import dataclass

from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException
from app.core.speech.env_utils import resolve_required_env

DEFAULT_VOLCENGINE_STT_ENDPOINT = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async"
DEFAULT_VOLCENGINE_STT_MAX_DURATION_SECONDS = 60


@dataclass(frozen=True)
class VolcengineSttConfig:
    """火山实时 STT 运行时配置。"""

    endpoint: str
    app_id: str
    access_token: str
    resource_id: str
    max_duration_seconds: int


def _parse_positive_int(*, value: str | None, name: str, default: int) -> int:
    if value is None or value.strip() == "":
        return default
    try:
        resolved = int(value)
    except ValueError as exc:
        raise ServiceException(
            code=ResponseCode.INTERNAL_ERROR,
            message=f"{name} must be an integer",
        ) from exc
    if resolved <= 0:
        raise ServiceException(
            code=ResponseCode.INTERNAL_ERROR,
            message=f"{name} must be greater than 0",
        )
    return resolved


def resolve_volcengine_stt_config() -> VolcengineSttConfig:
    """从环境变量解析实时 STT 配置。"""

    app_id = resolve_required_env("VOLCENGINE_STT_APP_ID")
    access_token = resolve_required_env("VOLCENGINE_STT_ACCESS_TOKEN")
    resource_id = resolve_required_env("VOLCENGINE_STT_RESOURCE_ID")
    endpoint = (os.getenv("VOLCENGINE_STT_ENDPOINT") or DEFAULT_VOLCENGINE_STT_ENDPOINT).strip()
    if not endpoint:
        endpoint = DEFAULT_VOLCENGINE_STT_ENDPOINT
    max_duration_seconds = _parse_positive_int(
        value=os.getenv("VOLCENGINE_STT_MAX_DURATION_SECONDS"),
        name="VOLCENGINE_STT_MAX_DURATION_SECONDS",
        default=DEFAULT_VOLCENGINE_STT_MAX_DURATION_SECONDS,
    )
    return VolcengineSttConfig(
        endpoint=endpoint,
        app_id=app_id,
        access_token=access_token,
        resource_id=resource_id,
        max_duration_seconds=max_duration_seconds,
    )


def build_volcengine_stt_headers(
        config: VolcengineSttConfig,
        *,
        connect_id: str | None = None,
) -> dict[str, str]:
    """构造连接火山实时 STT 所需 websocket 请求头。"""

    resolved_connect_id = (connect_id or str(uuid.uuid4())).strip()
    if not resolved_connect_id:
        resolved_connect_id = str(uuid.uuid4())
    return {
        "X-Api-App-Key": config.app_id,
        "X-Api-Access-Key": config.access_token,
        "X-Api-Resource-Id": config.resource_id,
        "X-Api-Connect-Id": resolved_connect_id,
    }


__all__ = [
    "DEFAULT_VOLCENGINE_STT_ENDPOINT",
    "DEFAULT_VOLCENGINE_STT_MAX_DURATION_SECONDS",
    "VolcengineSttConfig",
    "resolve_volcengine_stt_config",
    "build_volcengine_stt_headers",
]
