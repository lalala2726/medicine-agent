from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

import websockets
from loguru import logger

from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException
from app.core.speech.config import (
    VolcengineTtsConfig,
    build_volcengine_tts_headers,
    is_volcengine_tts_startup_connect_enabled,
    is_volcengine_tts_startup_fail_fast_enabled,
    resolve_volcengine_tts_config,
)
from app.core.speech.text_sanitizer import TtsTextSanitizer
from app.core.speech.volcengine_tts_protocol import (
    EventType,
    MsgType,
    finish_connection,
    finish_session,
    receive_message,
    start_connection,
    start_session,
    task_request,
    wait_for_event,
)
from app.schemas.document.message import MessageRole
from app.services.conversation_service import get_admin_conversation_by_id
from app.services.message_service import get_message_by_uuid

MAX_WS_MESSAGE_SIZE = 10 * 1024 * 1024
DEFAULT_TTS_TRUNCATION_PREFIX_TEMPLATE = "文本太多了，这边只读取前{max_chars}个字符。"

_AUDIO_MEDIA_TYPE_BY_ENCODING = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "pcm": "audio/L16",
    "ogg": "audio/ogg",
}


@dataclass(frozen=True)
class MessageTtsStream:
    """消息转语音的流式输出封装。"""

    audio_stream: AsyncIterator[bytes]
    media_type: str


def resolve_audio_media_type(encoding: str) -> str:
    """根据音频编码推导 HTTP `Content-Type`。"""

    normalized = encoding.strip().lower()
    if not normalized:
        return "application/octet-stream"
    return _AUDIO_MEDIA_TYPE_BY_ENCODING.get(normalized, "application/octet-stream")


def prepare_tts_text(
        *,
        raw_text: str,
        max_text_chars: int,
) -> str:
    """
    对待合成文本执行发送前处理。

    处理规则：
    1. 先做白名单清洗，仅保留可播报文本；
    2. 清洗后为空则拒绝转语音；
    3. 超过 `max_text_chars` 时，截取前 N 个字符并添加固定提示前缀。
    """

    sanitized_text = TtsTextSanitizer.sanitize_text(raw_text)
    if not sanitized_text:
        raise ServiceException(
            code=ResponseCode.BAD_REQUEST,
            message="消息清洗后为空，无法转语音",
        )

    if len(sanitized_text) <= max_text_chars:
        return sanitized_text

    truncation_prefix = DEFAULT_TTS_TRUNCATION_PREFIX_TEMPLATE.format(max_chars=max_text_chars).strip()
    truncated_text = sanitized_text[:max_text_chars]
    final_text = f"{truncation_prefix}{truncated_text}" if truncation_prefix else truncated_text

    logger.info(
        "Volcengine TTS input text truncated max_chars={max_chars} source_chars={source_chars} final_chars={final_chars}",
        max_chars=max_text_chars,
        source_chars=len(sanitized_text),
        final_chars=len(final_text),
    )
    return final_text


def _normalize_message_uuid(message_uuid: str) -> str:
    """标准化并校验消息 UUID。"""

    normalized = message_uuid.strip()
    if normalized:
        return normalized
    raise ServiceException(
        code=ResponseCode.BAD_REQUEST,
        message="message_uuid 不能为空",
    )


def _load_message_text_for_tts(
        *,
        message_uuid: str,
        user_id: int,
) -> str:
    """
    加载并校验待合成消息文本。

    校验规则：
    1. 消息存在；
    2. 消息所属会话属于当前用户，且为管理端会话；
    3. 仅允许 `role=ai`；
    4. 文本内容非空。
    """

    message = get_message_by_uuid(message_uuid)
    if message is None:
        raise ServiceException(
            code=ResponseCode.NOT_FOUND,
            message="消息不存在",
        )

    conversation = get_admin_conversation_by_id(
        conversation_id=message.conversation_id,
        user_id=user_id,
    )
    if conversation is None:
        raise ServiceException(
            code=ResponseCode.NOT_FOUND,
            message="消息不存在或无权限访问",
        )

    if message.role != MessageRole.AI:
        raise ServiceException(
            code=ResponseCode.BAD_REQUEST,
            message="仅支持 AI 消息转语音",
        )

    text = message.content.strip()
    if not text:
        raise ServiceException(
            code=ResponseCode.BAD_REQUEST,
            message="消息内容为空，无法转语音",
        )
    return text


def build_message_tts_stream(
        *,
        message_uuid: str,
        user_id: int,
) -> MessageTtsStream:
    """
    按 `message_uuid` 构建双向 TTS 音频流。

    说明：
    - 在返回流之前会先完成消息与权限校验；
    - 音色、编码、采样率全部由服务端环境变量控制；
    - 返回的异步迭代器会边接收上游音频边向下游输出分片。
    """

    normalized_message_uuid = _normalize_message_uuid(message_uuid)
    text = _load_message_text_for_tts(
        message_uuid=normalized_message_uuid,
        user_id=user_id,
    )
    config = resolve_volcengine_tts_config()
    prepared_text = prepare_tts_text(
        raw_text=text,
        max_text_chars=config.max_text_chars,
    )
    return MessageTtsStream(
        audio_stream=_stream_tts_audio(text=prepared_text, config=config),
        media_type=resolve_audio_media_type(config.encoding),
    )


def _build_start_session_payload(*, config: VolcengineTtsConfig) -> bytes:
    """构造 `StartSession` 事件请求体。"""

    payload = {
        "event": EventType.StartSession,
        "namespace": "BidirectionalTTS",
        "user": {
            "uid": str(uuid.uuid4()),
        },
        "req_params": {
            "speaker": config.voice_type,
            "audio_params": {
                "format": config.encoding,
                "sample_rate": config.sample_rate,
                "enable_timestamp": True,
            },
            "additions": json.dumps(
                {
                    "disable_markdown_filter": False,
                }
            ),
        },
    }
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _build_task_request_payload(*, config: VolcengineTtsConfig, text: str) -> bytes:
    """构造 `TaskRequest` 事件请求体。"""

    payload = {
        "event": EventType.TaskRequest,
        "namespace": "BidirectionalTTS",
        "req_params": {
            "speaker": config.voice_type,
            "text": text,
            "audio_params": {
                "format": config.encoding,
                "sample_rate": config.sample_rate,
                "enable_timestamp": True,
            },
        },
    }
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _decode_payload(payload: bytes) -> str:
    """将二进制载荷安全解码为文本，仅用于日志输出。"""

    try:
        return payload.decode("utf-8")
    except Exception:
        return ""


def _extract_log_id(websocket: object) -> str:
    """
    提取握手响应中的 `x-tt-logid`，便于启动日志追踪。

    不同 `websockets` 版本的响应对象结构略有差异，因此这里做防御式读取。
    """

    try:
        response = getattr(websocket, "response", None)
        if response is None:
            return ""
        headers = getattr(response, "headers", None)
        if headers is None:
            return ""
        getter = getattr(headers, "get", None)
        if callable(getter):
            return str(getter("x-tt-logid") or "")
        return ""
    except Exception:
        return ""


async def verify_volcengine_tts_connection_on_startup() -> None:
    """
    启动阶段执行一次火山双向 TTS 连接探活，并打印关键信息到控制台。

    行为说明：
    1. `VOLCENGINE_TTS_STARTUP_CONNECT_ENABLED=false` 时直接跳过；
    2. 配置不完整时打印提示并跳过；
    3. 配置完整时执行 websocket 握手与 `StartConnection/FinishConnection`；
    4. 失败时默认仅告警；若 `VOLCENGINE_TTS_STARTUP_FAIL_FAST=true` 则中断启动。
    """

    if not is_volcengine_tts_startup_connect_enabled():
        logger.info("Volcengine TTS startup connect is disabled.")
        return

    try:
        config = resolve_volcengine_tts_config()
    except ServiceException as exc:
        logger.warning(
            "Skip Volcengine TTS startup connect due to incomplete config: {message}",
            message=exc.message,
        )
        if is_volcengine_tts_startup_fail_fast_enabled():
            raise
        return

    websocket = None
    connect_id = str(uuid.uuid4())
    headers = build_volcengine_tts_headers(config, connect_id=connect_id)
    started_at = time.monotonic()

    logger.info(
        "Volcengine TTS startup connect begin endpoint={endpoint} connect_id={connect_id} resource_id={resource_id} voice_type={voice_type} encoding={encoding} sample_rate={sample_rate}",
        endpoint=config.endpoint,
        connect_id=connect_id,
        resource_id=config.resource_id,
        voice_type=config.voice_type,
        encoding=config.encoding,
        sample_rate=config.sample_rate,
    )

    try:
        websocket = await websockets.connect(
            config.endpoint,
            additional_headers=headers,
            max_size=MAX_WS_MESSAGE_SIZE,
        )
        log_id = _extract_log_id(websocket)
        logger.info(
            "Volcengine TTS websocket connected connect_id={connect_id} log_id={log_id}",
            connect_id=connect_id,
            log_id=log_id or "-",
        )

        await start_connection(websocket)
        await wait_for_event(
            websocket,
            msg_type=MsgType.FullServerResponse,
            event_type=EventType.ConnectionStarted,
        )

        elapsed_ms = int((time.monotonic() - started_at) * 1000)
        logger.info(
            "Volcengine TTS startup handshake success connect_id={connect_id} elapsed_ms={elapsed_ms}",
            connect_id=connect_id,
            elapsed_ms=elapsed_ms,
        )

        await finish_connection(websocket)
        await wait_for_event(
            websocket,
            msg_type=MsgType.FullServerResponse,
            event_type=EventType.ConnectionFinished,
        )
        logger.info(
            "Volcengine TTS startup connect closed connect_id={connect_id}",
            connect_id=connect_id,
        )
    except Exception as exc:
        logger.opt(exception=exc).warning(
            "Volcengine TTS startup connect failed connect_id={connect_id}",
            connect_id=connect_id,
        )
        if is_volcengine_tts_startup_fail_fast_enabled():
            raise ServiceException(
                code=ResponseCode.INTERNAL_ERROR,
                message="Volcengine TTS 启动连接失败",
            ) from exc
    finally:
        if websocket is not None:
            try:
                await websocket.close()
            except Exception:
                pass


async def _stream_tts_audio(
        *,
        text: str,
        config: VolcengineTtsConfig,
) -> AsyncIterator[bytes]:
    """
    建立到火山双向 TTS 的 WebSocket 并流式产出音频字节。

    生命周期：
    1. StartConnection -> ConnectionStarted
    2. StartSession -> SessionStarted
    3. TaskRequest + FinishSession
    4. 持续读取 `AudioOnlyServer` 直到 `SessionFinished`
    5. FinishConnection 并关闭 websocket
    """

    websocket = None
    session_id = str(uuid.uuid4())
    connect_id = str(uuid.uuid4())
    headers = build_volcengine_tts_headers(config, connect_id=connect_id)

    try:
        websocket = await websockets.connect(
            config.endpoint,
            additional_headers=headers,
            max_size=MAX_WS_MESSAGE_SIZE,
        )

        await start_connection(websocket)
        await wait_for_event(
            websocket,
            msg_type=MsgType.FullServerResponse,
            event_type=EventType.ConnectionStarted,
        )

        await start_session(
            websocket,
            _build_start_session_payload(config=config),
            session_id,
        )
        await wait_for_event(
            websocket,
            msg_type=MsgType.FullServerResponse,
            event_type=EventType.SessionStarted,
        )

        await task_request(
            websocket,
            _build_task_request_payload(config=config, text=text),
            session_id,
        )
        await finish_session(websocket, session_id)

        while True:
            message = await receive_message(websocket)
            if message.type == MsgType.AudioOnlyServer:
                if message.payload:
                    yield message.payload
                continue

            if message.type == MsgType.FullServerResponse:
                if message.event == EventType.SessionFinished:
                    break
                if message.event == EventType.SessionFailed:
                    logger.warning(
                        "Volcengine TTS session failed event={event} payload={payload}",
                        event=int(message.event),
                        payload=_decode_payload(message.payload),
                    )
                    break
                continue

            if message.type == MsgType.Error:
                logger.warning(
                    "Volcengine TTS protocol error code={code} payload={payload}",
                    code=message.error_code,
                    payload=_decode_payload(message.payload),
                )
                break

    except Exception as exc:
        logger.opt(exception=exc).warning(
            "Volcengine TTS streaming interrupted connect_id={connect_id} session_id={session_id}",
            connect_id=connect_id,
            session_id=session_id,
        )
    finally:
        if websocket is not None:
            try:
                await finish_connection(websocket)
                await wait_for_event(
                    websocket,
                    msg_type=MsgType.FullServerResponse,
                    event_type=EventType.ConnectionFinished,
                )
            except Exception:
                pass
            try:
                await websocket.close()
            except Exception:
                pass
