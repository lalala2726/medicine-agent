import asyncio
from types import SimpleNamespace

import pytest

from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException
from app.core.speech.config import VolcengineTtsConfig
from app.core.speech.volcengine_tts_protocol import EventType, Message, MsgType
from app.schemas.document.message import MessageRole
import app.core.speech.tts_client as service_module


class _DummyWebSocket:
    def __init__(self):
        self.sent_frames: list[bytes] = []
        self.closed = False

    async def send(self, frame: bytes):
        self.sent_frames.append(frame)

    async def close(self):
        self.closed = True


def _build_config() -> VolcengineTtsConfig:
    return VolcengineTtsConfig(
        endpoint="wss://example.com/tts",
        app_id="app-id",
        access_token="token",
        resource_id="volc.service_type.10029",
        voice_type="zh_female_1",
        encoding="mp3",
        sample_rate=24000,
        max_text_chars=300,
    )


async def _collect_stream(stream):
    chunks: list[bytes] = []
    async for item in stream:
        chunks.append(item)
    return chunks


def test_build_message_tts_stream_raises_not_found_when_message_missing(monkeypatch):
    monkeypatch.setattr(service_module, "get_message_by_uuid", lambda _uuid: None)

    with pytest.raises(ServiceException) as exc_info:
        service_module.build_message_tts_stream(
            message_uuid="msg-1",
            user_id=1,
        )

    assert exc_info.value.code == ResponseCode.NOT_FOUND.code


def test_build_message_tts_stream_raises_not_found_when_message_not_owned(monkeypatch):
    monkeypatch.setattr(
        service_module,
        "get_message_by_uuid",
        lambda _uuid: SimpleNamespace(
            conversation_id="507f1f77bcf86cd799439011",
            role=MessageRole.AI,
            content="hello",
        ),
    )
    monkeypatch.setattr(
        service_module,
        "get_admin_conversation_by_id",
        lambda **_kwargs: None,
    )

    with pytest.raises(ServiceException) as exc_info:
        service_module.build_message_tts_stream(
            message_uuid="msg-2",
            user_id=2,
        )

    assert exc_info.value.code == ResponseCode.NOT_FOUND.code


def test_build_message_tts_stream_raises_bad_request_when_message_role_not_ai(monkeypatch):
    monkeypatch.setattr(
        service_module,
        "get_message_by_uuid",
        lambda _uuid: SimpleNamespace(
            conversation_id="507f1f77bcf86cd799439011",
            role=MessageRole.USER,
            content="user text",
        ),
    )
    monkeypatch.setattr(
        service_module,
        "get_admin_conversation_by_id",
        lambda **_kwargs: SimpleNamespace(uuid="conv-3", user_id=3),
    )

    with pytest.raises(ServiceException) as exc_info:
        service_module.build_message_tts_stream(
            message_uuid="msg-3",
            user_id=3,
        )

    assert exc_info.value.code == ResponseCode.BAD_REQUEST.code


def test_prepare_tts_text_truncates_with_prefix():
    config = _build_config()
    config = VolcengineTtsConfig(
        endpoint=config.endpoint,
        app_id=config.app_id,
        access_token=config.access_token,
        resource_id=config.resource_id,
        voice_type=config.voice_type,
        encoding=config.encoding,
        sample_rate=config.sample_rate,
        max_text_chars=10,
    )

    raw_text = "###Hello**World** https://example.com 这是一段很长的测试文本"
    prepared = service_module.prepare_tts_text(
        raw_text=raw_text,
        max_text_chars=config.max_text_chars,
    )

    assert prepared.sent_text.startswith("文本太多了，这边只读取前10个字符。")
    assert "https://" not in prepared.sent_text.lower()
    assert "#" not in prepared.sent_text
    assert "*" not in prepared.sent_text
    assert prepared.is_truncated is True
    assert prepared.billable_chars == len(prepared.sent_text)


def test_prepare_tts_text_raises_when_sanitized_empty():
    config = _build_config()
    raw_text = "```json\n{\"a\":1}\n```\nhttps://example.com\n"

    with pytest.raises(ServiceException) as exc_info:
        service_module.prepare_tts_text(
            raw_text=raw_text,
            max_text_chars=config.max_text_chars,
        )

    assert exc_info.value.code == ResponseCode.BAD_REQUEST.code
    assert "清洗后为空" in exc_info.value.message


def test_stream_message_tts_yields_audio_chunks_in_order(monkeypatch):
    monkeypatch.setattr(
        service_module,
        "get_message_by_uuid",
        lambda _uuid: SimpleNamespace(
            conversation_id="507f1f77bcf86cd799439011",
            role=MessageRole.AI,
            content="测试文本",
        ),
    )
    monkeypatch.setattr(
        service_module,
        "get_admin_conversation_by_id",
        lambda **_kwargs: SimpleNamespace(uuid="conv-4", user_id=4),
    )
    monkeypatch.setattr(
        service_module,
        "resolve_volcengine_tts_config",
        lambda **_kwargs: _build_config(),
    )
    monkeypatch.setattr(
        service_module,
        "build_volcengine_tts_headers",
        lambda *_args, **_kwargs: {"X-Api-App-Key": "app-id"},
    )
    usage_calls: list[dict] = []
    monkeypatch.setattr(
        service_module,
        "add_message_tts_usage",
        lambda **kwargs: usage_calls.append(kwargs) or "507f1f77bcf86cd799439071",
    )

    dummy_ws = _DummyWebSocket()
    dummy_ws.response = SimpleNamespace(headers={"x-tt-logid": "log-success"})

    async def _fake_connect(*_args, **_kwargs):
        return dummy_ws

    monkeypatch.setattr(service_module.websockets, "connect", _fake_connect)

    async def _noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(service_module, "start_connection", _noop)
    monkeypatch.setattr(service_module, "start_session", _noop)
    monkeypatch.setattr(service_module, "task_request", _noop)
    monkeypatch.setattr(service_module, "finish_session", _noop)
    monkeypatch.setattr(service_module, "finish_connection", _noop)
    monkeypatch.setattr(service_module, "wait_for_event", _noop)

    received_messages = [
        Message(type=MsgType.AudioOnlyServer, payload=b"chunk-1"),
        Message(type=MsgType.AudioOnlyServer, payload=b"chunk-2"),
        Message(type=MsgType.FullServerResponse, event=EventType.SessionFinished, payload=b"{}"),
    ]

    async def _fake_receive_message(*_args, **_kwargs):
        return received_messages.pop(0)

    monkeypatch.setattr(service_module, "receive_message", _fake_receive_message)

    stream = service_module.build_message_tts_stream(
        message_uuid="msg-4",
        user_id=4,
    )
    chunks = asyncio.run(_collect_stream(stream.audio_stream))

    assert chunks == [b"chunk-1", b"chunk-2"]
    assert dummy_ws.closed is True
    assert len(usage_calls) == 1
    assert usage_calls[0]["message_uuid"] == "msg-4"
    assert usage_calls[0]["conversation_uuid"] == "conv-4"
    assert usage_calls[0]["user_id"] == 4
    assert usage_calls[0]["sent_text"] == "测试文本"
    assert usage_calls[0]["source_text_chars"] == len("测试文本")
    assert usage_calls[0]["sanitized_text_chars"] == len("测试文本")
    assert usage_calls[0]["is_truncated"] is False
    assert usage_calls[0]["audio_chunk_count"] == 2
    assert usage_calls[0]["audio_bytes"] == len(b"chunk-1") + len(b"chunk-2")
    assert usage_calls[0]["provider_log_id"] == "log-success"


def test_stream_message_tts_stops_gracefully_when_upstream_interrupted(monkeypatch):
    monkeypatch.setattr(
        service_module,
        "get_message_by_uuid",
        lambda _uuid: SimpleNamespace(
            conversation_id="507f1f77bcf86cd799439011",
            role=MessageRole.AI,
            content="测试文本",
        ),
    )
    monkeypatch.setattr(
        service_module,
        "get_admin_conversation_by_id",
        lambda **_kwargs: SimpleNamespace(uuid="conv-5", user_id=5),
    )
    monkeypatch.setattr(
        service_module,
        "resolve_volcengine_tts_config",
        lambda **_kwargs: _build_config(),
    )
    monkeypatch.setattr(
        service_module,
        "build_volcengine_tts_headers",
        lambda *_args, **_kwargs: {"X-Api-App-Key": "app-id"},
    )
    usage_calls: list[dict] = []
    monkeypatch.setattr(
        service_module,
        "add_message_tts_usage",
        lambda **kwargs: usage_calls.append(kwargs) or "507f1f77bcf86cd799439072",
    )

    dummy_ws = _DummyWebSocket()

    async def _fake_connect(*_args, **_kwargs):
        return dummy_ws

    monkeypatch.setattr(service_module.websockets, "connect", _fake_connect)

    async def _noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(service_module, "start_connection", _noop)
    monkeypatch.setattr(service_module, "start_session", _noop)
    monkeypatch.setattr(service_module, "task_request", _noop)
    monkeypatch.setattr(service_module, "finish_session", _noop)
    monkeypatch.setattr(service_module, "finish_connection", _noop)
    monkeypatch.setattr(service_module, "wait_for_event", _noop)

    async def _raise_on_receive(*_args, **_kwargs):
        raise RuntimeError("connection closed")

    monkeypatch.setattr(service_module, "receive_message", _raise_on_receive)

    stream = service_module.build_message_tts_stream(
        message_uuid="msg-5",
        user_id=5,
    )
    chunks = asyncio.run(_collect_stream(stream.audio_stream))

    assert chunks == []
    assert dummy_ws.closed is True
    assert usage_calls == []


def test_stream_message_tts_persists_usage_with_truncated_flag(monkeypatch):
    config = _build_config()
    config = VolcengineTtsConfig(
        endpoint=config.endpoint,
        app_id=config.app_id,
        access_token=config.access_token,
        resource_id=config.resource_id,
        voice_type=config.voice_type,
        encoding=config.encoding,
        sample_rate=config.sample_rate,
        max_text_chars=5,
    )
    monkeypatch.setattr(
        service_module,
        "get_message_by_uuid",
        lambda _uuid: SimpleNamespace(
            conversation_id="507f1f77bcf86cd799439011",
            role=MessageRole.AI,
            content="这是很长很长的文本，用于触发截断。",
        ),
    )
    monkeypatch.setattr(
        service_module,
        "get_admin_conversation_by_id",
        lambda **_kwargs: SimpleNamespace(uuid="conv-6", user_id=6),
    )
    monkeypatch.setattr(
        service_module,
        "resolve_volcengine_tts_config",
        lambda **_kwargs: config,
    )
    monkeypatch.setattr(
        service_module,
        "build_volcengine_tts_headers",
        lambda *_args, **_kwargs: {"X-Api-App-Key": "app-id"},
    )
    usage_calls: list[dict] = []
    monkeypatch.setattr(
        service_module,
        "add_message_tts_usage",
        lambda **kwargs: usage_calls.append(kwargs) or "507f1f77bcf86cd799439073",
    )

    dummy_ws = _DummyWebSocket()

    async def _fake_connect(*_args, **_kwargs):
        return dummy_ws

    monkeypatch.setattr(service_module.websockets, "connect", _fake_connect)

    async def _noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(service_module, "start_connection", _noop)
    monkeypatch.setattr(service_module, "start_session", _noop)
    monkeypatch.setattr(service_module, "task_request", _noop)
    monkeypatch.setattr(service_module, "finish_session", _noop)
    monkeypatch.setattr(service_module, "finish_connection", _noop)
    monkeypatch.setattr(service_module, "wait_for_event", _noop)

    received_messages = [
        Message(type=MsgType.AudioOnlyServer, payload=b"chunk"),
        Message(type=MsgType.FullServerResponse, event=EventType.SessionFinished, payload=b"{}"),
    ]

    async def _fake_receive_message(*_args, **_kwargs):
        return received_messages.pop(0)

    monkeypatch.setattr(service_module, "receive_message", _fake_receive_message)

    stream = service_module.build_message_tts_stream(
        message_uuid="msg-6",
        user_id=6,
    )
    chunks = asyncio.run(_collect_stream(stream.audio_stream))

    assert chunks == [b"chunk"]
    assert len(usage_calls) == 1
    assert usage_calls[0]["message_uuid"] == "msg-6"
    assert usage_calls[0]["is_truncated"] is True
    assert usage_calls[0]["max_text_chars"] == 5
    assert usage_calls[0]["sent_text"].startswith("文本太多了，这边只读取前5个字符。")
    assert usage_calls[0]["sanitized_text_chars"] >= 5


def test_verify_volcengine_tts_connection_on_startup_success(monkeypatch):
    monkeypatch.setattr(
        service_module,
        "is_volcengine_tts_startup_connect_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        service_module,
        "is_volcengine_tts_startup_fail_fast_enabled",
        lambda: False,
    )
    monkeypatch.setattr(
        service_module,
        "resolve_volcengine_tts_config",
        lambda: _build_config(),
    )
    monkeypatch.setattr(
        service_module,
        "build_volcengine_tts_headers",
        lambda *_args, **_kwargs: {"X-Api-App-Key": "app-id"},
    )

    dummy_ws = _DummyWebSocket()
    dummy_ws.response = SimpleNamespace(headers={"x-tt-logid": "log-1"})

    async def _fake_connect(*_args, **_kwargs):
        return dummy_ws

    call_events: list[tuple[str, object]] = []

    async def _fake_start_connection(*_args, **_kwargs):
        call_events.append(("start_connection", None))

    async def _fake_finish_connection(*_args, **_kwargs):
        call_events.append(("finish_connection", None))

    async def _fake_wait_for_event(*_args, **kwargs):
        call_events.append(("wait_for_event", kwargs.get("event_type")))
        return Message(type=MsgType.FullServerResponse, event=kwargs.get("event_type"), payload=b"{}")

    monkeypatch.setattr(service_module.websockets, "connect", _fake_connect)
    monkeypatch.setattr(service_module, "start_connection", _fake_start_connection)
    monkeypatch.setattr(service_module, "finish_connection", _fake_finish_connection)
    monkeypatch.setattr(service_module, "wait_for_event", _fake_wait_for_event)

    asyncio.run(service_module.verify_volcengine_tts_connection_on_startup())

    assert ("start_connection", None) in call_events
    assert ("finish_connection", None) in call_events
    assert ("wait_for_event", EventType.ConnectionStarted) in call_events
    assert ("wait_for_event", EventType.ConnectionFinished) in call_events
    assert dummy_ws.closed is True


def test_verify_volcengine_tts_connection_on_startup_skips_when_disabled(monkeypatch):
    monkeypatch.setattr(
        service_module,
        "is_volcengine_tts_startup_connect_enabled",
        lambda: False,
    )

    called = {"connect": False}

    async def _fake_connect(*_args, **_kwargs):
        called["connect"] = True
        return _DummyWebSocket()

    monkeypatch.setattr(service_module.websockets, "connect", _fake_connect)

    asyncio.run(service_module.verify_volcengine_tts_connection_on_startup())

    assert called["connect"] is False
