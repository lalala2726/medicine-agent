import pytest

from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException
from app.core.speech import env_utils as speech_env_utils
from app.core.speech.stt import config as stt_config_module


def _set_required_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOLCENGINE_APP_ID", "test-app-id")
    monkeypatch.setenv("VOLCENGINE_ACCESS_TOKEN", "test-access-token")
    monkeypatch.setenv("VOLCENGINE_STT_RESOURCE_ID", "volc.seedasr.sauc.duration")


def _disable_dotenv_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(speech_env_utils, "_read_dotenv_value", lambda _name: "")


def test_resolve_volcengine_stt_config_uses_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_dotenv_lookup(monkeypatch)
    _set_required_env(monkeypatch)
    monkeypatch.delenv("VOLCENGINE_STT_ENDPOINT", raising=False)
    monkeypatch.delenv("VOLCENGINE_STT_MAX_DURATION_SECONDS", raising=False)

    config = stt_config_module.resolve_volcengine_stt_config()

    assert config.endpoint == stt_config_module.DEFAULT_VOLCENGINE_STT_ENDPOINT
    assert config.max_duration_seconds == stt_config_module.DEFAULT_VOLCENGINE_STT_MAX_DURATION_SECONDS
    assert config.app_id == "test-app-id"
    assert config.access_token == "test-access-token"
    assert config.resource_id == "volc.seedasr.sauc.duration"


def test_resolve_volcengine_stt_config_raises_when_required_env_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_dotenv_lookup(monkeypatch)
    monkeypatch.delenv("VOLCENGINE_APP_ID", raising=False)
    monkeypatch.delenv("VOLCENGINE_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("VOLCENGINE_STT_RESOURCE_ID", raising=False)

    with pytest.raises(ServiceException) as exc_info:
        stt_config_module.resolve_volcengine_stt_config()

    assert exc_info.value.code == ResponseCode.INTERNAL_ERROR.code
    assert "VOLCENGINE_APP_ID" in exc_info.value.message


def test_resolve_volcengine_stt_config_raises_when_max_duration_invalid(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_dotenv_lookup(monkeypatch)
    _set_required_env(monkeypatch)
    monkeypatch.setenv("VOLCENGINE_STT_MAX_DURATION_SECONDS", "abc")

    with pytest.raises(ServiceException) as exc_info:
        stt_config_module.resolve_volcengine_stt_config()

    assert exc_info.value.code == ResponseCode.INTERNAL_ERROR.code
    assert "VOLCENGINE_STT_MAX_DURATION_SECONDS" in exc_info.value.message


def test_resolve_volcengine_stt_config_caps_max_duration_to_internal_upper_bound(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_dotenv_lookup(monkeypatch)
    _set_required_env(monkeypatch)
    monkeypatch.setenv("VOLCENGINE_STT_MAX_DURATION_SECONDS", "1800")

    config = stt_config_module.resolve_volcengine_stt_config()

    assert config.max_duration_seconds == stt_config_module.MAX_VOLCENGINE_STT_MAX_DURATION_SECONDS


def test_resolve_volcengine_stt_config_supports_custom_max_duration_from_env(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_dotenv_lookup(monkeypatch)
    _set_required_env(monkeypatch)
    monkeypatch.setenv("VOLCENGINE_STT_MAX_DURATION_SECONDS", "300")

    config = stt_config_module.resolve_volcengine_stt_config()

    assert config.max_duration_seconds == 300


def test_build_volcengine_stt_headers_contains_required_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_dotenv_lookup(monkeypatch)
    _set_required_env(monkeypatch)
    config = stt_config_module.resolve_volcengine_stt_config()

    headers = stt_config_module.build_volcengine_stt_headers(config, connect_id="connect-1")

    assert headers == {
        "X-Api-App-Key": "test-app-id",
        "X-Api-Access-Key": "test-access-token",
        "X-Api-Resource-Id": "volc.seedasr.sauc.duration",
        "X-Api-Connect-Id": "connect-1",
    }
