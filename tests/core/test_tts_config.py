import pytest

from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException
from app.core.speech import env_utils as speech_env_utils
from app.core.speech.tts import config as tts_config_module


def _set_shared_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOLCENGINE_APP_ID", "test-app-id")
    monkeypatch.setenv("VOLCENGINE_ACCESS_TOKEN", "test-access-token")


def _disable_dotenv_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(speech_env_utils, "_read_dotenv_value", lambda _name: "")


def test_resolve_volcengine_tts_config_uses_shared_auth_and_defaults(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_dotenv_lookup(monkeypatch)
    _set_shared_env(monkeypatch)
    monkeypatch.setenv("VOLCENGINE_TTS_VOICE_TYPE", "zh_female_xiaohe_uranus_bigtts")
    monkeypatch.delenv("VOLCENGINE_TTS_ENDPOINT", raising=False)
    monkeypatch.delenv("VOLCENGINE_TTS_ENCODING", raising=False)
    monkeypatch.delenv("VOLCENGINE_TTS_SAMPLE_RATE", raising=False)
    monkeypatch.delenv("VOLCENGINE_TTS_MAX_TEXT_CHARS", raising=False)
    monkeypatch.delenv("VOLCENGINE_TTS_RESOURCE_ID", raising=False)

    config = tts_config_module.resolve_volcengine_tts_config()

    assert config.app_id == "test-app-id"
    assert config.access_token == "test-access-token"
    assert config.endpoint == tts_config_module.DEFAULT_VOLCENGINE_TTS_ENDPOINT
    assert config.encoding == tts_config_module.DEFAULT_VOLCENGINE_TTS_ENCODING
    assert config.sample_rate == tts_config_module.DEFAULT_VOLCENGINE_TTS_SAMPLE_RATE
    assert config.max_text_chars == tts_config_module.DEFAULT_VOLCENGINE_TTS_MAX_TEXT_CHARS
    assert config.resource_id == "volc.service_type.10029"


def test_resolve_volcengine_tts_config_raises_when_shared_env_missing(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_dotenv_lookup(monkeypatch)
    monkeypatch.delenv("VOLCENGINE_APP_ID", raising=False)
    monkeypatch.delenv("VOLCENGINE_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv("VOLCENGINE_TTS_VOICE_TYPE", "zh_female_xiaohe_uranus_bigtts")

    with pytest.raises(ServiceException) as exc_info:
        tts_config_module.resolve_volcengine_tts_config()

    assert exc_info.value.code == ResponseCode.INTERNAL_ERROR.code
    assert "VOLCENGINE_APP_ID" in exc_info.value.message


def test_resolve_volcengine_tts_config_keeps_tts_specific_settings(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_dotenv_lookup(monkeypatch)
    _set_shared_env(monkeypatch)
    monkeypatch.setenv("VOLCENGINE_TTS_VOICE_TYPE", "S_demo_voice")
    monkeypatch.setenv("VOLCENGINE_TTS_RESOURCE_ID", "seed-tts-2.0")
    monkeypatch.setenv("VOLCENGINE_TTS_ENDPOINT", "wss://tts.example/ws")
    monkeypatch.setenv("VOLCENGINE_TTS_ENCODING", "wav")
    monkeypatch.setenv("VOLCENGINE_TTS_SAMPLE_RATE", "16000")
    monkeypatch.setenv("VOLCENGINE_TTS_MAX_TEXT_CHARS", "500")

    config = tts_config_module.resolve_volcengine_tts_config()

    assert config.endpoint == "wss://tts.example/ws"
    assert config.resource_id == "seed-tts-2.0"
    assert config.encoding == "wav"
    assert config.sample_rate == 16000
    assert config.max_text_chars == 500
    assert config.voice_type == "S_demo_voice"
