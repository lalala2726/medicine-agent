from __future__ import annotations

from app.agent.assistant.model_switch import model_switch


def test_model_switch_returns_simple_model_and_think_disabled() -> None:
    """测试目的：验证 simple 难度的模型路由；预期结果：返回 qwen-flash 且 think=False。"""

    model_name, think = model_switch({"routing": {"task_difficulty": "simple"}})

    assert model_name == "qwen-flash"
    assert think is False


def test_model_switch_returns_normal_model_and_think_disabled() -> None:
    """测试目的：验证 normal 难度的模型路由；预期结果：返回 qwen-max 且 think=False。"""

    model_name, think = model_switch({"routing": {"task_difficulty": "normal"}})

    assert model_name == "qwen-max"
    assert think is False


def test_model_switch_returns_complex_model_and_think_enabled() -> None:
    """测试目的：验证 complex 难度的模型路由；预期结果：返回 qwen3-plus 且 think=True。"""

    model_name, think = model_switch({"routing": {"task_difficulty": "complex"}})

    assert model_name == "qwen3-plus"
    assert think is True


def test_model_switch_falls_back_to_default_when_difficulty_missing() -> None:
    """测试目的：验证缺失或未知难度时的兜底行为；预期结果：返回 qwen-max 且 think=False。"""

    model_name_missing, think_missing = model_switch({})
    model_name_unknown, think_unknown = model_switch({"routing": {"task_difficulty": "other"}})

    assert model_name_missing == "qwen-max"
    assert think_missing is False
    assert model_name_unknown == "qwen-max"
    assert think_unknown is False
