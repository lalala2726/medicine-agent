from __future__ import annotations

import json
from typing import Any

import pytest

from app.core.config_sync import snapshot as agent_config_module


class _FakeRedis:
    def __init__(self, return_value: Any = None, error: Exception | None = None) -> None:
        self.return_value = return_value
        self.error = error
        self.get_calls: list[str] = []

    def get(self, key: str) -> Any:
        self.get_calls.append(key)
        if self.error is not None:
            raise self.error
        return self.return_value


@pytest.fixture(autouse=True)
def _clear_agent_config_state() -> None:
    agent_config_module.clear_agent_config_snapshot_state()
    yield
    agent_config_module.clear_agent_config_snapshot_state()


def _build_snapshot_payload(*, route_model_name: str = "gpt-4.1-mini") -> dict[str, Any]:
    return {
        "updatedAt": "2026-03-11T14:30:00+08:00",
        "updatedBy": "admin",
        "knowledgeBase": {
            "embeddingDim": 1024,
            "embeddingModel": {
                "reasoningEnabled": False,
                "maxTokens": None,
                "temperature": None,
                "model": {
                    "provider": "Qwen",
                    "model": "text-embedding-v4",
                    "modelType": "EMBEDDING",
                    "baseUrl": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "apiKey": "sk-embed",
                    "supportReasoning": False,
                    "supportVision": False,
                },
            },
        },
        "adminAssistant": {
            "routeModel": {
                "reasoningEnabled": False,
                "maxTokens": 1024,
                "temperature": 0.0,
                "model": {
                    "provider": "OpenAI",
                    "model": route_model_name,
                    "modelType": "CHAT",
                    "baseUrl": "https://api.openai.com/v1",
                    "apiKey": "sk-route",
                    "supportReasoning": True,
                    "supportVision": False,
                },
            },
            "chatModel": {
                "reasoningEnabled": True,
                "maxTokens": 4096,
                "temperature": 0.7,
                "model": {
                    "provider": "OpenAI",
                    "model": "gpt-4.1",
                    "modelType": "CHAT",
                    "baseUrl": "https://api.openai.com/v1",
                    "apiKey": "sk-chat",
                    "supportReasoning": True,
                    "supportVision": True,
                },
            },
        },
        "imageRecognition": {
            "imageRecognitionModel": {
                "reasoningEnabled": True,
                "maxTokens": 2048,
                "temperature": 0.2,
                "model": {
                    "provider": "Qwen",
                    "model": "qwen-vl",
                    "modelType": "CHAT",
                    "baseUrl": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "apiKey": "sk-image",
                    "supportReasoning": True,
                    "supportVision": True,
                },
            },
        },
        "chatHistorySummary": {
            "chatHistorySummaryModel": {
                "reasoningEnabled": False,
                "maxTokens": 4096,
                "temperature": 0.3,
                "model": {
                    "provider": "OpenAI",
                    "model": "gpt-summary",
                    "modelType": "CHAT",
                    "baseUrl": "https://api.openai.com/v1",
                    "apiKey": "sk-summary",
                    "supportReasoning": True,
                    "supportVision": False,
                },
            },
        },
        "chatTitle": {
            "chatTitleModel": {
                "reasoningEnabled": False,
                "maxTokens": 32,
                "temperature": 0.2,
                "model": {
                    "provider": "OpenAI",
                    "model": "gpt-title",
                    "modelType": "CHAT",
                    "baseUrl": "https://api.openai.com/v1",
                    "apiKey": "sk-title",
                    "supportReasoning": True,
                    "supportVision": False,
                },
            },
        },
        "speech": {
            "provider": "volcengine",
            "appId": "speech-app-id",
            "accessToken": "speech-access-token",
            "speechRecognition": {
                "resourceId": "volc.seedasr.sauc.duration",
            },
            "textToSpeech": {
                "resourceId": "seed-tts-2.0",
                "voiceType": "zh_female_xiaohe_uranus_bigtts",
                "maxTextChars": 300,
            },
        },
    }


def test_initialize_agent_config_snapshot_loads_valid_redis_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """测试目的：启动时 Redis 有合法配置时应成功加载并归一化 provider；预期结果：图片识别与聊天历史总结等新字段均可正常解析。"""

    payload = json.dumps(_build_snapshot_payload()).encode("utf-8")
    fake_redis = _FakeRedis(return_value=payload)
    monkeypatch.setattr(agent_config_module, "get_redis_connection", lambda: fake_redis)

    snapshot = agent_config_module.initialize_agent_config_snapshot()

    assert snapshot.admin_assistant is not None
    assert snapshot.admin_assistant.route_model is not None
    assert snapshot.admin_assistant.route_model.model is not None
    assert snapshot.admin_assistant.route_model.model.provider == "openai"
    assert snapshot.image_recognition is not None
    assert snapshot.image_recognition.image_recognition_model is not None
    assert snapshot.chat_history_summary is not None
    assert snapshot.chat_history_summary.chat_history_summary_model is not None
    assert snapshot.chat_title is not None
    assert snapshot.chat_title.chat_title_model is not None
    assert snapshot.get_speech_shared_auth() == ("speech-app-id", "speech-access-token")
    assert snapshot.get_speech_stt_resource_id() == "volc.seedasr.sauc.duration"
    assert snapshot.get_speech_tts_resource_id() == "seed-tts-2.0"
    assert snapshot.get_speech_tts_voice_type() == "zh_female_xiaohe_uranus_bigtts"
    assert snapshot.get_speech_tts_max_text_chars() == 300
    assert snapshot.knowledge_base is not None
    assert snapshot.knowledge_base.embedding_model is not None
    assert snapshot.knowledge_base.embedding_model.model is not None
    assert snapshot.knowledge_base.embedding_model.model.provider == "aliyun"
    assert fake_redis.get_calls == [agent_config_module.AGENT_CONFIG_REDIS_KEY]


def test_initialize_agent_config_snapshot_falls_back_to_local_when_redis_missing(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试目的：首次启动 Redis 无数据时应回退本地快照；预期结果：不抛异常且写入本地兜底标记。"""

    monkeypatch.setattr(agent_config_module, "get_redis_connection", lambda: _FakeRedis(return_value=None))

    snapshot = agent_config_module.initialize_agent_config_snapshot()

    assert snapshot.updated_by == "local_env_fallback"


def test_initialize_agent_config_snapshot_accepts_payload_without_removed_config_version(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试目的：迁移后 Redis 配置不再携带 configVersion；预期结果：仍可成功加载快照。"""

    payload_dict = _build_snapshot_payload()
    payload = json.dumps(payload_dict).encode("utf-8")
    monkeypatch.setattr(agent_config_module, "get_redis_connection", lambda: _FakeRedis(return_value=payload))

    snapshot = agent_config_module.initialize_agent_config_snapshot()

    assert snapshot.admin_assistant is not None
    assert not hasattr(snapshot, "config_version")


def test_initialize_agent_config_snapshot_unwraps_wrapped_data_root(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试目的：当前 Redis 可能存在 `data` 包装层；预期结果：应自动解包并正确读取语音与其他业务字段。"""

    payload_dict = {
        "@class": "java.util.LinkedHashMap",
        "data": _build_snapshot_payload(),
    }
    payload = json.dumps(payload_dict).encode("utf-8")
    monkeypatch.setattr(agent_config_module, "get_redis_connection", lambda: _FakeRedis(return_value=payload))

    snapshot = agent_config_module.initialize_agent_config_snapshot()

    assert snapshot.admin_assistant is not None
    assert snapshot.get_speech_shared_auth() == ("speech-app-id", "speech-access-token")
    assert snapshot.get_speech_stt_resource_id() == "volc.seedasr.sauc.duration"
    assert snapshot.get_speech_tts_resource_id() == "seed-tts-2.0"
    assert snapshot.get_speech_tts_voice_type() == "zh_female_xiaohe_uranus_bigtts"
    assert snapshot.get_speech_tts_max_text_chars() == 300


def test_initialize_agent_config_snapshot_accepts_chinese_provider_aliases(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试目的：管理端可能写入中文 provider 名称；预期结果：应正确归一化为内部 provider。"""

    payload_dict = _build_snapshot_payload()
    payload_dict["adminAssistant"]["chatModel"]["model"]["provider"] = "阿里云百联"
    payload_dict["adminAssistant"]["routeModel"]["model"]["provider"] = "火山引擎"
    payload = json.dumps(payload_dict).encode("utf-8")
    monkeypatch.setattr(agent_config_module, "get_redis_connection", lambda: _FakeRedis(return_value=payload))

    snapshot = agent_config_module.initialize_agent_config_snapshot()

    assert snapshot.admin_assistant is not None
    assert snapshot.admin_assistant.chat_model is not None
    assert snapshot.admin_assistant.chat_model.model is not None
    assert snapshot.admin_assistant.chat_model.model.provider == "aliyun"
    assert snapshot.admin_assistant.route_model is not None
    assert snapshot.admin_assistant.route_model.model is not None
    assert snapshot.admin_assistant.route_model.model.provider == "volcengine"


def test_refresh_agent_config_snapshot_keeps_previous_redis_snapshot_on_failure(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试目的：已有成功 Redis 快照后刷新失败不应回退覆盖；预期结果：继续保留旧快照内容。"""

    initial_payload = json.dumps(_build_snapshot_payload(route_model_name="gpt-old")).encode("utf-8")
    monkeypatch.setattr(agent_config_module, "get_redis_connection", lambda: _FakeRedis(return_value=initial_payload))
    agent_config_module.initialize_agent_config_snapshot()

    monkeypatch.setattr(
        agent_config_module,
        "get_redis_connection",
        lambda: _FakeRedis(error=RuntimeError("redis down")),
    )

    refreshed = agent_config_module.refresh_agent_config_snapshot(
        redis_key=agent_config_module.AGENT_CONFIG_REDIS_KEY,
    )

    current_snapshot = agent_config_module.get_current_agent_config_snapshot()
    assert refreshed is False
    assert current_snapshot.admin_assistant is not None
    assert current_snapshot.admin_assistant.route_model is not None
    assert current_snapshot.admin_assistant.route_model.model is not None
    assert current_snapshot.admin_assistant.route_model.model.model == "gpt-old"


def test_refresh_agent_config_snapshot_always_reload_when_notification_arrives(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试目的：收到通知后不再进行版本门控；预期结果：即使版本未变化也会重新读取 Redis。"""

    initial_payload = json.dumps(_build_snapshot_payload(route_model_name="gpt-route-same")).encode("utf-8")
    initial_redis = _FakeRedis(return_value=initial_payload)
    monkeypatch.setattr(agent_config_module, "get_redis_connection", lambda: initial_redis)
    agent_config_module.initialize_agent_config_snapshot()

    reloaded_payload = json.dumps(_build_snapshot_payload(route_model_name="gpt-route-same")).encode("utf-8")
    reloaded_redis = _FakeRedis(return_value=reloaded_payload)
    monkeypatch.setattr(agent_config_module, "get_redis_connection", lambda: reloaded_redis)

    refreshed = agent_config_module.refresh_agent_config_snapshot(
        redis_key=agent_config_module.AGENT_CONFIG_REDIS_KEY,
    )

    assert refreshed is True
    assert reloaded_redis.get_calls == [agent_config_module.AGENT_CONFIG_REDIS_KEY]


def test_refresh_agent_config_snapshot_applies_redis_payload(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试目的：收到刷新通知后应替换本地快照；预期结果：当前快照更新为 Redis 最新内容。"""

    initial_payload = json.dumps(_build_snapshot_payload(route_model_name="gpt-old")).encode("utf-8")
    monkeypatch.setattr(agent_config_module, "get_redis_connection", lambda: _FakeRedis(return_value=initial_payload))
    agent_config_module.initialize_agent_config_snapshot()

    refreshed_payload = json.dumps(_build_snapshot_payload(route_model_name="gpt-new")).encode("utf-8")
    monkeypatch.setattr(agent_config_module, "get_redis_connection", lambda: _FakeRedis(return_value=refreshed_payload))

    refreshed = agent_config_module.refresh_agent_config_snapshot(
        redis_key=agent_config_module.AGENT_CONFIG_REDIS_KEY,
    )

    current_snapshot = agent_config_module.get_current_agent_config_snapshot()
    assert refreshed is True
    assert current_snapshot.admin_assistant is not None
    assert current_snapshot.admin_assistant.route_model is not None
    assert current_snapshot.admin_assistant.route_model.model is not None
    assert current_snapshot.admin_assistant.route_model.model.model == "gpt-new"


def test_initialize_agent_config_snapshot_accepts_partial_speech_auth_without_failing(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试目的：Redis 里语音鉴权只给半套时不应拖垮整份快照；预期结果：快照可成功加载，但共享鉴权 getter 返回空。"""

    payload_dict = _build_snapshot_payload()
    payload_dict["speech"]["accessToken"] = None
    payload = json.dumps(payload_dict).encode("utf-8")
    monkeypatch.setattr(agent_config_module, "get_redis_connection", lambda: _FakeRedis(return_value=payload))

    snapshot = agent_config_module.initialize_agent_config_snapshot()

    assert snapshot.speech is not None
    assert snapshot.speech.app_id == "speech-app-id"
    assert snapshot.get_speech_shared_auth() is None


def test_refresh_agent_config_snapshot_applies_latest_speech_values(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试目的：收到刷新通知后语音配置也应同步更新；预期结果：当前快照可读到新的音色与最大字符数。"""

    initial_payload = json.dumps(_build_snapshot_payload()).encode("utf-8")
    monkeypatch.setattr(agent_config_module, "get_redis_connection", lambda: _FakeRedis(return_value=initial_payload))
    agent_config_module.initialize_agent_config_snapshot()

    refreshed_payload_dict = _build_snapshot_payload()
    refreshed_payload_dict["speech"]["textToSpeech"]["resourceId"] = "seed-tts-3.0"
    refreshed_payload_dict["speech"]["textToSpeech"]["voiceType"] = "S_demo_voice"
    refreshed_payload_dict["speech"]["textToSpeech"]["maxTextChars"] = 512
    refreshed_payload = json.dumps(refreshed_payload_dict).encode("utf-8")
    monkeypatch.setattr(agent_config_module, "get_redis_connection", lambda: _FakeRedis(return_value=refreshed_payload))

    refreshed = agent_config_module.refresh_agent_config_snapshot(
        redis_key=agent_config_module.AGENT_CONFIG_REDIS_KEY,
    )

    current_snapshot = agent_config_module.get_current_agent_config_snapshot()
    assert refreshed is True
    assert current_snapshot.get_speech_tts_resource_id() == "seed-tts-3.0"
    assert current_snapshot.get_speech_tts_voice_type() == "S_demo_voice"
    assert current_snapshot.get_speech_tts_max_text_chars() == 512
