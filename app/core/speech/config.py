from __future__ import annotations

import os
import uuid
from dataclasses import dataclass

from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException

DEFAULT_VOLCENGINE_TTS_ENDPOINT = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"
DEFAULT_VOLCENGINE_TTS_ENCODING = "mp3"
DEFAULT_VOLCENGINE_TTS_SAMPLE_RATE = 24000
DEFAULT_VOLCENGINE_TTS_MAX_TEXT_CHARS = 300
DEFAULT_VOLCENGINE_TTS_STARTUP_CONNECT_ENABLED = True
DEFAULT_VOLCENGINE_TTS_STARTUP_FAIL_FAST = False


@dataclass(frozen=True)
class VolcengineTtsConfig:
    """火山双向 TTS 运行时配置。"""

    endpoint: str
    app_id: str
    access_token: str
    resource_id: str
    voice_type: str
    encoding: str
    sample_rate: int
    max_text_chars: int


def infer_resource_id(voice_type: str) -> str:
    """
    按官方示例规则推导 `resource_id`。

    规则：
    - `voice_type` 以 `S_` 开头时：`volc.megatts.default`
    - 其他情况：`volc.service_type.10029`
    """

    if voice_type.startswith("S_"):
        return "volc.megatts.default"
    return "volc.service_type.10029"


def _resolve_required_env(name: str) -> str:
    """读取必填环境变量；为空时抛出配置异常。"""

    value = (os.getenv(name) or "").strip()
    if value:
        return value
    raise ServiceException(
        code=ResponseCode.INTERNAL_ERROR,
        message=f"{name} is not set",
    )


def _parse_positive_int(*, value: str | None, name: str, default: int) -> int:
    """解析正整数配置；空值返回默认值。"""

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


def _parse_bool(*, value: str | None, default: bool) -> bool:
    """解析布尔环境变量。"""

    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized == "":
        return default
    return normalized in {"1", "true", "yes", "on"}


def resolve_volcengine_tts_config() -> VolcengineTtsConfig:
    """
    从环境变量解析火山双向 TTS 配置。

    说明：
    - 前端仅传 `message_uuid`，音色/编码/采样率全部由服务端配置控制；
    - `VOLCENGINE_TTS_RESOURCE_ID` 为空时，根据音色自动推导；
    - 文本最大字符数由 `VOLCENGINE_TTS_MAX_TEXT_CHARS` 控制。
    """

    app_id = _resolve_required_env("VOLCENGINE_TTS_APP_ID")
    access_token = _resolve_required_env("VOLCENGINE_TTS_ACCESS_TOKEN")
    endpoint = (os.getenv("VOLCENGINE_TTS_ENDPOINT") or DEFAULT_VOLCENGINE_TTS_ENDPOINT).strip()
    if not endpoint:
        endpoint = DEFAULT_VOLCENGINE_TTS_ENDPOINT

    resolved_voice_type = (os.getenv("VOLCENGINE_TTS_VOICE_TYPE") or "").strip()
    if not resolved_voice_type:
        raise ServiceException(
            code=ResponseCode.BAD_REQUEST,
            message="voice_type 不能为空",
        )

    resolved_encoding = (
            os.getenv("VOLCENGINE_TTS_ENCODING")
            or DEFAULT_VOLCENGINE_TTS_ENCODING
    ).strip().lower()
    if not resolved_encoding:
        resolved_encoding = DEFAULT_VOLCENGINE_TTS_ENCODING

    resolved_sample_rate = _parse_positive_int(
        value=os.getenv("VOLCENGINE_TTS_SAMPLE_RATE"),
        name="VOLCENGINE_TTS_SAMPLE_RATE",
        default=DEFAULT_VOLCENGINE_TTS_SAMPLE_RATE,
    )
    resolved_max_text_chars = _parse_positive_int(
        value=os.getenv("VOLCENGINE_TTS_MAX_TEXT_CHARS"),
        name="VOLCENGINE_TTS_MAX_TEXT_CHARS",
        default=DEFAULT_VOLCENGINE_TTS_MAX_TEXT_CHARS,
    )

    configured_resource_id = (os.getenv("VOLCENGINE_TTS_RESOURCE_ID") or "").strip()
    resource_id = configured_resource_id or infer_resource_id(resolved_voice_type)

    return VolcengineTtsConfig(
        endpoint=endpoint,
        app_id=app_id,
        access_token=access_token,
        resource_id=resource_id,
        voice_type=resolved_voice_type,
        encoding=resolved_encoding,
        sample_rate=resolved_sample_rate,
        max_text_chars=resolved_max_text_chars,
    )


def is_volcengine_tts_startup_connect_enabled() -> bool:
    """是否在服务启动阶段执行 TTS 连接探活。"""

    return _parse_bool(
        value=os.getenv("VOLCENGINE_TTS_STARTUP_CONNECT_ENABLED"),
        default=DEFAULT_VOLCENGINE_TTS_STARTUP_CONNECT_ENABLED,
    )


def is_volcengine_tts_startup_fail_fast_enabled() -> bool:
    """启动探活失败时是否中断服务启动。"""

    return _parse_bool(
        value=os.getenv("VOLCENGINE_TTS_STARTUP_FAIL_FAST"),
        default=DEFAULT_VOLCENGINE_TTS_STARTUP_FAIL_FAST,
    )


def build_volcengine_tts_headers(
        config: VolcengineTtsConfig,
        *,
        connect_id: str | None = None,
) -> dict[str, str]:
    """构造连接火山双向 TTS 所需的 WebSocket 请求头。"""

    resolved_connect_id = (connect_id or str(uuid.uuid4())).strip()
    if not resolved_connect_id:
        resolved_connect_id = str(uuid.uuid4())

    return {
        "X-Api-App-Key": config.app_id,
        "X-Api-Access-Key": config.access_token,
        "X-Api-Resource-Id": config.resource_id,
        "X-Api-Connect-Id": resolved_connect_id,
    }
