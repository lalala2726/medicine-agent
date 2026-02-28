from __future__ import annotations

import json
import struct
import time
import uuid
from dataclasses import dataclass
from typing import Any

import websockets
from loguru import logger

from app.core.exception.exceptions import ServiceException
from app.core.speech.stt.config import (
    VolcengineSttConfig,
    build_volcengine_stt_headers,
    resolve_volcengine_stt_config,
)
from app.core.speech.volcengine_tts_protocol import (
    CompressionBits,
    MsgType,
    MsgTypeFlagBits,
    SerializationBits,
    SttServerMessage,
    compress_payload,
    parse_stt_server_message,
)

MAX_WS_MESSAGE_SIZE = 10 * 1024 * 1024
DEFAULT_SEGMENT_MILLIS = 200
DEFAULT_AUDIO_FORMAT = "pcm"
DEFAULT_AUDIO_CODEC = "raw"
DEFAULT_AUDIO_RATE = 16000
DEFAULT_AUDIO_BITS = 16
DEFAULT_AUDIO_CHANNEL = 1


@dataclass(frozen=True)
class SttStartRequest:
    """前端 `start` 指令映射后的 STT 请求参数。"""

    enable_itn: bool = True
    enable_punc: bool = True
    show_utterances: bool = True
    result_type: str = "single"


class VolcengineSttClient:
    """火山实时 STT websocket 客户端（bigmodel_async）。"""

    def __init__(self, *, config: VolcengineSttConfig):
        self._config = config
        self._sequence = 1
        self._connect_id = str(uuid.uuid4())
        self._websocket: Any | None = None
        self._provider_log_id = ""

    @property
    def provider_log_id(self) -> str:
        return self._provider_log_id

    async def connect(self) -> None:
        if self._websocket is not None:
            return
        self._websocket = await websockets.connect(
            self._config.endpoint,
            additional_headers=build_volcengine_stt_headers(
                self._config,
                connect_id=self._connect_id,
            ),
            max_size=MAX_WS_MESSAGE_SIZE,
        )
        self._provider_log_id = self._extract_log_id(self._websocket)

    async def close(self) -> None:
        websocket = self._websocket
        self._websocket = None
        if websocket is None:
            return
        try:
            await websocket.close()
        except Exception:
            pass

    async def send_full_client_request(
            self,
            *,
            request: SttStartRequest,
            user_id: int | None = None,
    ) -> None:
        payload = {
            "user": {
                "uid": str(user_id or "unknown"),
            },
            "audio": {
                "format": DEFAULT_AUDIO_FORMAT,
                "codec": DEFAULT_AUDIO_CODEC,
                "rate": DEFAULT_AUDIO_RATE,
                "bits": DEFAULT_AUDIO_BITS,
                "channel": DEFAULT_AUDIO_CHANNEL,
            },
            "request": {
                "model_name": "bigmodel",
                "enable_itn": request.enable_itn,
                "enable_punc": request.enable_punc,
                "show_utterances": request.show_utterances,
                "result_type": request.result_type,
            },
        }
        frame = self._build_client_frame(
            message_type=MsgType.FullClientRequest,
            flag=MsgTypeFlagBits.PositiveSeq,
            serialization=SerializationBits.JSON,
            compression=CompressionBits.Gzip,
            sequence=self._consume_sequence(),
            payload=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        )
        await self._send_frame(frame)

    async def send_audio_chunk(self, chunk: bytes, *, is_last: bool) -> None:
        flag = MsgTypeFlagBits.NegativeSeq if is_last else MsgTypeFlagBits.PositiveSeq
        sequence = self._consume_sequence()
        if is_last:
            sequence = -sequence
        frame = self._build_client_frame(
            message_type=MsgType.AudioOnlyClient,
            flag=flag,
            serialization=SerializationBits.Raw,
            compression=CompressionBits.Gzip,
            sequence=sequence,
            payload=chunk,
        )
        await self._send_frame(frame)

    async def receive_server_message(self) -> SttServerMessage:
        if self._websocket is None:
            raise RuntimeError("stt websocket is not connected")
        message = await self._websocket.recv()
        if isinstance(message, str):
            raise ValueError(f"unexpected text websocket frame: {message}")
        if not isinstance(message, bytes):
            raise ValueError(f"unexpected websocket frame type: {type(message)}")
        return parse_stt_server_message(message)

    def _consume_sequence(self) -> int:
        sequence = self._sequence
        self._sequence += 1
        return sequence

    async def _send_frame(self, frame: bytes) -> None:
        if self._websocket is None:
            raise RuntimeError("stt websocket is not connected")
        await self._websocket.send(frame)

    @staticmethod
    def _build_client_frame(
            *,
            message_type: MsgType,
            flag: MsgTypeFlagBits,
            serialization: SerializationBits,
            compression: CompressionBits,
            sequence: int,
            payload: bytes,
    ) -> bytes:
        header = bytes(
            [
                0x11,  # version=1, header_size=1 (4 bytes)
                ((int(message_type) & 0x0F) << 4) | (int(flag) & 0x0F),
                ((int(serialization) & 0x0F) << 4) | (int(compression) & 0x0F),
                0x00,
            ]
        )
        compressed_payload = compress_payload(payload, compression)
        frame = bytearray()
        frame.extend(header)
        frame.extend(struct.pack(">i", sequence))
        frame.extend(struct.pack(">I", len(compressed_payload)))
        frame.extend(compressed_payload)
        return bytes(frame)

    @staticmethod
    def _extract_log_id(websocket: object) -> str:
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


def _extract_startup_connect_error_detail(exc: Exception) -> tuple[int | None, str | None, str]:
    """
    从 websocket 握手异常中提取可读错误信息，避免启动日志只看到长堆栈。
    """

    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    reason_phrase = getattr(response, "reason_phrase", None)
    body = getattr(response, "body", None)
    body_text = ""
    if isinstance(body, bytes):
        try:
            body_text = body.decode("utf-8", errors="ignore")
        except Exception:
            body_text = ""
    elif body is not None:
        body_text = str(body)
    if len(body_text) > 500:
        body_text = body_text[:500] + "..."
    return status_code, reason_phrase, body_text


async def verify_volcengine_stt_connection_on_startup() -> None:
    """
    启动阶段执行一次火山实时 STT 连接探活，并打印关键信息到控制台。

    行为说明：
    1. 配置不完整时打印失败原因并跳过；
    2. 配置完整时执行 websocket 握手探活；
    3. 成功/失败均打印日志；
    4. 不抛异常中断应用启动。
    """

    try:
        config = resolve_volcengine_stt_config()
    except ServiceException as exc:
        logger.warning(
            "Volcengine STT startup connect failed due to invalid config: {message}",
            message=exc.message,
        )
        return

    client = VolcengineSttClient(config=config)
    started_at = time.monotonic()
    logger.info(
        "Volcengine STT startup connect begin endpoint={endpoint} resource_id={resource_id} max_duration_seconds={max_duration_seconds}",
        endpoint=config.endpoint,
        resource_id=config.resource_id,
        max_duration_seconds=config.max_duration_seconds,
    )
    try:
        await client.connect()
        elapsed_ms = int((time.monotonic() - started_at) * 1000)
        logger.info(
            "Volcengine STT startup connect success log_id={log_id} elapsed_ms={elapsed_ms}",
            log_id=client.provider_log_id or "-",
            elapsed_ms=elapsed_ms,
        )
    except Exception as exc:
        status_code, reason_phrase, body_text = _extract_startup_connect_error_detail(exc)
        if status_code is not None:
            logger.warning(
                "Volcengine STT startup connect failed status={status_code} reason={reason_phrase} body={body_text}",
                status_code=status_code,
                reason_phrase=reason_phrase or "-",
                body_text=body_text or "-",
            )
        else:
            logger.opt(exception=exc).warning("Volcengine STT startup connect failed")
    finally:
        await client.close()


__all__ = [
    "DEFAULT_AUDIO_BITS",
    "DEFAULT_AUDIO_CHANNEL",
    "DEFAULT_AUDIO_CODEC",
    "DEFAULT_AUDIO_FORMAT",
    "DEFAULT_AUDIO_RATE",
    "DEFAULT_SEGMENT_MILLIS",
    "MAX_WS_MESSAGE_SIZE",
    "SttStartRequest",
    "VolcengineSttClient",
    "verify_volcengine_stt_connection_on_startup",
]
