from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

gateway_node = importlib.import_module("app.agent.assistant.node.gateway_node")


def _install_gateway_runtime_mocks(monkeypatch: pytest.MonkeyPatch, *, captured: dict) -> None:
    """
    功能描述：
        安装 gateway_router 运行所需的最小 mock，避免依赖真实模型请求。

    参数说明：
        monkeypatch (pytest.MonkeyPatch): pytest monkeypatch 工具实例。
        captured (dict): 用于捕获 create_chat_model 与 trace_item 的断言容器。

    返回值：
        None

    异常说明：
        无。
    """

    fake_llm = object()
    fake_agent = object()

    def _fake_create_chat_model(**kwargs):
        captured["create_chat_model_kwargs"] = kwargs
        return fake_llm

    def _fake_create_agent(**kwargs):
        captured["create_agent_kwargs"] = kwargs
        return fake_agent

    monkeypatch.setattr(gateway_node, "create_chat_model", _fake_create_chat_model)
    monkeypatch.setattr(gateway_node, "create_agent", _fake_create_agent)
    monkeypatch.setattr(
        gateway_node,
        "agent_invoke",
        lambda _agent, _history_messages: SimpleNamespace(payload={}, content=""),
    )
    monkeypatch.setattr(
        gateway_node,
        "record_agent_trace",
        lambda **_kwargs: {
            "raw_content": {"route_target": "chat_agent", "task_difficulty": "normal"},
            "text": '{"route_target":"chat_agent","task_difficulty":"normal"}',
            "model_name": "trace-model-name",
            "is_usage_complete": True,
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )

    def _fake_append_trace_and_refresh_token_usage(_execution_traces, trace_item):
        captured["trace_item"] = trace_item
        return [trace_item], {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}

    monkeypatch.setattr(
        gateway_node,
        "append_trace_and_refresh_token_usage",
        _fake_append_trace_and_refresh_token_usage,
    )


def test_gateway_router_uses_dedicated_model_for_aliyun(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试目的：验证 aliyun 已配置专用路由模型时优先使用；预期结果：create_chat_model.model 与 execution_trace.model_name 均为专用模型。"""

    captured: dict[str, object] = {}
    monkeypatch.setenv("LLM_PROVIDER", "aliyun")
    monkeypatch.setenv("DASHSCOPE_GATEWAY_ROUTER_MODEL", "qwen-router-fast")
    monkeypatch.setenv("DASHSCOPE_CHAT_MODEL", "qwen-chat-fallback")
    _install_gateway_runtime_mocks(monkeypatch, captured=captured)

    result = gateway_node.gateway_router({"history_messages": [], "execution_traces": []})

    assert result["routing"]["route_target"] == "chat_agent"
    assert result["routing"]["task_difficulty"] == "normal"
    assert captured["create_chat_model_kwargs"]["model"] == "qwen-router-fast"
    assert captured["trace_item"]["model_name"] == "qwen-router-fast"


def test_gateway_router_falls_back_to_chat_model_when_dedicated_missing_for_aliyun(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试目的：验证 aliyun 未配置专用路由模型时回退通用 chat 模型；预期结果：create_chat_model.model 为 DASHSCOPE_CHAT_MODEL。"""

    captured: dict[str, object] = {}
    monkeypatch.setenv("LLM_PROVIDER", "aliyun")
    monkeypatch.delenv("DASHSCOPE_GATEWAY_ROUTER_MODEL", raising=False)
    monkeypatch.setenv("DASHSCOPE_CHAT_MODEL", "qwen-chat-fallback")
    _install_gateway_runtime_mocks(monkeypatch, captured=captured)

    gateway_node.gateway_router({"history_messages": [], "execution_traces": []})

    assert captured["create_chat_model_kwargs"]["model"] == "qwen-chat-fallback"
    assert captured["trace_item"]["model_name"] == "qwen-chat-fallback"


def test_resolve_gateway_router_model_name_raises_when_both_volcengine_keys_missing(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试目的：验证 volcengine 专用与回退模型都缺失时抛错；预期结果：RuntimeError 信息包含两个配置键名。"""

    monkeypatch.setenv("LLM_PROVIDER", "volcengine")
    monkeypatch.delenv("VOLCENGINE_LLM_GATEWAY_ROUTER_MODEL", raising=False)
    monkeypatch.delenv("VOLCENGINE_LLM_CHAT_MODEL", raising=False)

    with pytest.raises(
            RuntimeError,
            match="VOLCENGINE_LLM_GATEWAY_ROUTER_MODEL and VOLCENGINE_LLM_CHAT_MODEL are not set",
    ):
        gateway_node._resolve_gateway_router_model_name()


def test_gateway_router_execution_trace_uses_openai_dedicated_model_name(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试目的：验证 openai 配置专用路由模型时追踪稳定写入该模型名；预期结果：execution_trace.model_name 等于 OPENAI_GATEWAY_ROUTER_MODEL。"""

    captured: dict[str, object] = {}
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_GATEWAY_ROUTER_MODEL", "gpt-4o-mini-router")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-4o-mini-fallback")
    _install_gateway_runtime_mocks(monkeypatch, captured=captured)

    gateway_node.gateway_router({"history_messages": [], "execution_traces": []})

    assert captured["create_chat_model_kwargs"]["model"] == "gpt-4o-mini-router"
    assert captured["trace_item"]["model_name"] == "gpt-4o-mini-router"
