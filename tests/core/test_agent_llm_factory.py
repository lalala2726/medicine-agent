from __future__ import annotations

from typing import Any

import pytest

from app.agent.assistant.model_switch import model_switch
from app.core.config_sync import AgentChatModelSlot, AgentConfigSnapshot
from app.core.config_sync import llm as llm_factory


def _build_snapshot() -> AgentConfigSnapshot:
    return AgentConfigSnapshot.model_validate(
        {
            "updatedAt": "2026-03-11T14:30:00+08:00",
            "updatedBy": "admin",
            "knowledgeBase": {
                "embeddingModel": {
                    "model": {
                        "provider": "Qwen",
                        "model": "redis-embedding-model",
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
                    "reasoningEnabled": True,
                    "maxTokens": 4096,
                    "temperature": 0.2,
                    "model": {
                        "provider": "OpenAI",
                        "model": "gpt-route-redis",
                        "modelType": "CHAT",
                        "baseUrl": "https://api.openai.com/v1",
                        "apiKey": "sk-route",
                        "supportReasoning": True,
                        "supportVision": False,
                    },
                },
            },
            "imageRecognition": {
                "imageRecognitionModel": {
                    "reasoningEnabled": True,
                    "maxTokens": 2048,
                    "temperature": 0.3,
                    "model": {
                        "provider": "Qwen",
                        "model": "qwen-vl-redis",
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
                    "temperature": 0.1,
                    "model": {
                        "provider": "OpenAI",
                        "model": "gpt-summary-redis",
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
                        "model": "gpt-title-redis",
                        "modelType": "CHAT",
                        "baseUrl": "https://api.openai.com/v1",
                        "apiKey": "sk-title",
                        "supportReasoning": True,
                        "supportVision": False,
                    },
                },
            },
        },
    )


def test_create_agent_chat_llm_prefers_redis_slot_over_local_defaults(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试目的：聊天包装工厂应优先使用 Redis 槽位参数；预期结果：temperature/think/max_tokens 与运行时模型参数均来自 Redis。"""

    captured: dict[str, Any] = {}
    monkeypatch.setattr(llm_factory, "get_current_agent_config_snapshot", _build_snapshot)
    monkeypatch.setattr(llm_factory, "create_chat_model", lambda **kwargs: captured.update(kwargs) or "llm")

    result = llm_factory.create_agent_chat_llm(
        slot=AgentChatModelSlot.ROUTE,
        temperature=1.0,
        think=False,
        max_tokens=512,
    )

    assert result == "llm"
    assert captured["model"] == "gpt-route-redis"
    assert captured["provider"] == "openai"
    assert captured["base_url"] == "https://api.openai.com/v1"
    assert captured["api_key"] == "sk-route"
    assert captured["temperature"] == 0.2
    assert captured["max_tokens"] == 4096
    assert captured["think"] is True


def test_create_agent_chat_llm_uses_gateway_env_fallback_when_slot_missing(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试目的：路由槽位缺失时应回退 gateway 专用环境模型名；预期结果：透传 env 解析出的 model。"""

    captured: dict[str, Any] = {}
    empty_snapshot = AgentConfigSnapshot()
    monkeypatch.setattr(llm_factory, "get_current_agent_config_snapshot", lambda: empty_snapshot)
    monkeypatch.setattr(llm_factory, "_resolve_gateway_router_fallback_model_name", lambda: "gateway-env-model")
    monkeypatch.setattr(llm_factory, "create_chat_model", lambda **kwargs: captured.update(kwargs) or "llm")

    result = llm_factory.create_agent_chat_llm(
        slot=AgentChatModelSlot.ROUTE,
        temperature=0.0,
        think=False,
    )

    assert result == "llm"
    assert captured["model"] == "gateway-env-model"
    assert captured["temperature"] == 0.0
    assert captured["think"] is False


def test_create_agent_chat_llm_treats_zero_max_tokens_as_unlimited(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试目的：本地传入 max_tokens=0 时应视为不限制；预期结果：不会向底层透传 max_tokens。"""

    captured: dict[str, Any] = {}
    empty_snapshot = AgentConfigSnapshot()
    monkeypatch.setattr(llm_factory, "get_current_agent_config_snapshot", lambda: empty_snapshot)
    monkeypatch.setattr(llm_factory, "_resolve_gateway_router_fallback_model_name", lambda: "gateway-env-model")
    monkeypatch.setattr(llm_factory, "create_chat_model", lambda **kwargs: captured.update(kwargs) or "llm")

    result = llm_factory.create_agent_chat_llm(
        slot=AgentChatModelSlot.ROUTE,
        temperature=0.0,
        think=False,
        max_tokens=0,
    )

    assert result == "llm"
    assert "max_tokens" not in captured


def test_create_agent_image_llm_prefers_redis_slot_over_local_defaults(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试目的：图片识别包装工厂应优先使用 Redis 槽位参数；预期结果：运行时模型与温度/思考配置均来自 Redis。"""

    captured: dict[str, Any] = {}
    monkeypatch.setattr(llm_factory, "get_current_agent_config_snapshot", _build_snapshot)
    monkeypatch.setattr(llm_factory, "create_image_model", lambda **kwargs: captured.update(kwargs) or "image-llm")

    result = llm_factory.create_agent_image_llm(
        temperature=1.0,
        think=False,
        max_tokens=256,
    )

    assert result == "image-llm"
    assert captured["model"] == "qwen-vl-redis"
    assert captured["provider"] == "aliyun"
    assert captured["temperature"] == 0.3
    assert captured["max_tokens"] == 2048
    assert captured["think"] is True


def test_create_agent_summary_llm_prefers_redis_slot_over_local_defaults(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试目的：聊天历史总结包装工厂应优先使用 Redis 槽位参数；预期结果：模型与温度/token 配置均来自 Redis。"""

    captured: dict[str, Any] = {}
    monkeypatch.setattr(llm_factory, "get_current_agent_config_snapshot", _build_snapshot)
    monkeypatch.setattr(llm_factory, "create_chat_model", lambda **kwargs: captured.update(kwargs) or "summary-llm")

    result = llm_factory.create_agent_summary_llm(
        temperature=0.9,
        think=True,
        max_tokens=256,
    )

    assert result == "summary-llm"
    assert captured["model"] == "gpt-summary-redis"
    assert captured["provider"] == "openai"
    assert captured["temperature"] == 0.1
    assert captured["max_tokens"] == 4096
    assert captured["think"] is False


def test_create_agent_title_llm_prefers_redis_slot_over_local_defaults(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试目的：聊天标题包装工厂应优先使用 Redis 槽位参数；预期结果：模型与温度/token 配置均来自 Redis。"""

    captured: dict[str, Any] = {}
    monkeypatch.setattr(llm_factory, "get_current_agent_config_snapshot", _build_snapshot)
    monkeypatch.setattr(llm_factory, "create_chat_model", lambda **kwargs: captured.update(kwargs) or "title-llm")

    result = llm_factory.create_agent_title_llm(
        temperature=1.0,
        think=True,
        max_tokens=64,
    )

    assert result == "title-llm"
    assert captured["model"] == "gpt-title-redis"
    assert captured["provider"] == "openai"
    assert captured["temperature"] == 0.2
    assert captured["max_tokens"] == 32
    assert captured["think"] is False


def test_resolve_agent_summary_helpers_fall_back_to_env_when_slot_missing(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试目的：聊天历史总结模型名和预算解析在 Redis 缺失时应回退本地环境配置；预期结果：分别返回本地 summary env 模型名和 token 上限。"""

    empty_snapshot = AgentConfigSnapshot()
    monkeypatch.setattr(llm_factory, "get_current_agent_config_snapshot", lambda: empty_snapshot)
    monkeypatch.setattr(llm_factory, "_resolve_summary_fallback_model_name", lambda: "summary-env-model")
    monkeypatch.setattr(llm_factory, "_resolve_summary_fallback_max_tokens", lambda: 2048)

    assert llm_factory.resolve_agent_summary_model_name() == "summary-env-model"
    assert llm_factory.resolve_agent_summary_max_tokens() == 2048


def test_summary_slot_zero_max_tokens_means_unlimited(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试目的：聊天历史总结槽位 maxTokens=0 时应视为不限制；预期结果：不向底层透传 max_tokens，预算解析返回 None。"""

    snapshot = AgentConfigSnapshot.model_validate(
        {
            "chatHistorySummary": {
                "chatHistorySummaryModel": {
                    "reasoningEnabled": False,
                    "maxTokens": 0,
                    "temperature": 0.1,
                    "model": {
                        "provider": "OpenAI",
                        "model": "gpt-summary-redis",
                        "modelType": "CHAT",
                        "baseUrl": "https://api.openai.com/v1",
                        "apiKey": "sk-summary",
                        "supportReasoning": True,
                        "supportVision": False,
                    },
                },
            },
        },
    )
    captured: dict[str, Any] = {}
    monkeypatch.setattr(llm_factory, "get_current_agent_config_snapshot", lambda: snapshot)
    monkeypatch.setattr(llm_factory, "create_chat_model", lambda **kwargs: captured.update(kwargs) or "summary-llm")

    result = llm_factory.create_agent_summary_llm(max_tokens=128)

    assert result == "summary-llm"
    assert "max_tokens" not in captured
    assert llm_factory.resolve_agent_summary_max_tokens() is None


def test_create_agent_embedding_client_prefers_explicit_model_name(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试目的：向量包装工厂应优先使用显式传入模型名；预期结果：model 使用显式值，provider/base_url/api_key 仍来自 Redis。"""

    captured: dict[str, Any] = {}
    monkeypatch.setattr(llm_factory, "get_current_agent_config_snapshot", _build_snapshot)
    monkeypatch.setattr(
        llm_factory,
        "create_embedding_model",
        lambda **kwargs: captured.update(kwargs) or "embedding-client",
    )

    result = llm_factory.create_agent_embedding_client(
        model="remote-embedding-model",
        dimensions=1536,
    )

    assert result == "embedding-client"
    assert captured["model"] == "remote-embedding-model"
    assert captured["provider"] == "aliyun"
    assert captured["base_url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert captured["api_key"] == "sk-embed"
    assert captured["dimensions"] == 1536


def test_model_switch_returns_complex_slot_for_high_difficulty() -> None:
    """测试目的：业务难度为 high 时应选择 complex 槽位；预期结果：返回 `businessNodeComplexModel`。"""

    slot = model_switch({"routing": {"task_difficulty": "high"}})

    assert slot is AgentChatModelSlot.BUSINESS_COMPLEX
