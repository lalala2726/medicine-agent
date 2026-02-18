from app.schemas.admin_message import TokenUsage
from app.services import token_usage_service as service_module


def test_build_token_usage_uses_prompt_completion_sum():
    """验证 build_token_usage：总量以 prompt+completion 为准，忽略冲突 total_tokens。"""

    usage = service_module.build_token_usage(
        prompt_tokens=2,
        completion_tokens=3,
        total_tokens=99,
    )

    assert usage.prompt_tokens == 2
    assert usage.completion_tokens == 3
    assert usage.total_tokens == 5


def test_normalize_token_usage_fills_unknown_model_name_in_breakdown():
    """验证 normalize_token_usage：breakdown 缺失 model_name 时回退为 unknown。"""

    usage = service_module.normalize_token_usage(
        {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "breakdown": [
                {
                    "node_name": "chat_agent",
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                }
            ],
        }
    )

    assert usage is not None
    assert usage.breakdown is not None
    assert usage.breakdown[0].node_name == "chat_agent"
    assert usage.breakdown[0].model_name == "unknown"


def test_merge_assistant_token_usage_uses_stream_and_fills_missing_prompt(monkeypatch):
    """验证 merge_assistant_token_usage：流式缺失字段可用估算补齐，并保留 breakdown。"""

    monkeypatch.setattr(
        service_module,
        "estimate_prompt_completion_usage",
        lambda **_kwargs: TokenUsage(
            prompt_tokens=11,
            completion_tokens=7,
            total_tokens=18,
            breakdown=None,
        ),
    )

    result = service_module.merge_assistant_token_usage(
        stream_token_usage={
            "prompt_tokens": 0,
            "completion_tokens": 7,
            "total_tokens": 7,
            "breakdown": [
                {
                    "node_name": "chat_agent",
                    "model_name": "qwen-max",
                    "prompt_tokens": 0,
                    "completion_tokens": 7,
                    "total_tokens": 7,
                }
            ],
        },
        prompt_text="问题",
        completion_text="回答",
    )

    assert result is not None
    assert result.prompt_tokens == 11
    assert result.completion_tokens == 7
    assert result.total_tokens == 18
    assert result.breakdown is not None
    assert result.breakdown[0].model_name == "qwen-max"


def test_merge_assistant_token_usage_returns_none_when_all_sources_missing(monkeypatch):
    """验证 merge_assistant_token_usage：流式与估算都不可用时返回 None。"""

    monkeypatch.setattr(
        service_module,
        "estimate_prompt_completion_usage",
        lambda **_kwargs: None,
    )

    result = service_module.merge_assistant_token_usage(
        stream_token_usage=None,
        prompt_text="问题",
        completion_text="回答",
    )

    assert result is None


def test_merge_assistant_token_usage_returns_none_when_stream_zero_and_estimate_fails(monkeypatch):
    """验证 merge_assistant_token_usage：全 0 流式 usage 且估算失败时返回 None。"""

    monkeypatch.setattr(
        service_module,
        "estimate_prompt_completion_usage",
        lambda **_kwargs: None,
    )

    result = service_module.merge_assistant_token_usage(
        stream_token_usage={
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "breakdown": None,
        },
        prompt_text="问题",
        completion_text="回答",
    )

    assert result is None
