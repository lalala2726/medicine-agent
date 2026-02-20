from __future__ import annotations

from app.agent.admin.model_policy import (
    DEFAULT_NODE_GOAL,
    apply_model_profile_to_routing,
    build_gateway_decision,
    build_supervisor_decision,
    normalize_task_difficulty,
    resolve_model_profile,
)


def test_normalize_task_difficulty_fallbacks_to_normal():
    assert normalize_task_difficulty("simple") == "simple"
    assert normalize_task_difficulty("complex") == "complex"
    assert normalize_task_difficulty("简单") == "simple"
    assert normalize_task_difficulty("未知") == "normal"
    assert normalize_task_difficulty(None) == "normal"


def test_resolve_model_profile_mapping():
    assert resolve_model_profile("simple") == {
        "task_difficulty": "simple",
        "model": "qwen-flash",
        "think": False,
    }
    assert resolve_model_profile("normal") == {
        "task_difficulty": "normal",
        "model": "qwen-plus",
        "think": False,
    }
    assert resolve_model_profile("complex") == {
        "task_difficulty": "complex",
        "model": "qwen-max",
        "think": True,
    }
    assert resolve_model_profile("complex", allow_thinking=False)["think"] is False


def test_build_gateway_decision_with_fallbacks():
    route_target, task_difficulty = build_gateway_decision(
        {"route_target": "order_agent", "task_difficulty": "simple"}
    )
    assert route_target == "order_agent"
    assert task_difficulty == "simple"

    route_target, task_difficulty = build_gateway_decision(
        {"route_target": "unknown", "task_difficulty": "super-complex"}
    )
    assert route_target == "supervisor_agent"
    assert task_difficulty == "normal"


def test_build_supervisor_decision_with_fallbacks():
    target_node, node_goal, task_difficulty = build_supervisor_decision(
        {
            "target_node": "product_agent",
            "node_goal": "查询商品2001",
            "task_difficulty": "complex",
        },
        fallback_task_difficulty="normal",
    )
    assert target_node == "product_agent"
    assert node_goal == "查询商品2001"
    assert task_difficulty == "complex"

    target_node, node_goal, task_difficulty = build_supervisor_decision(
        {"target_node": "bad", "node_goal": "ignore"},
        fallback_task_difficulty="simple",
    )
    assert target_node == "summary_agent"
    assert node_goal == "ignore"
    assert task_difficulty == "simple"

    target_node, node_goal, task_difficulty = build_supervisor_decision(
        {"target_node": "summary_agent", "node_goal": ""},
        fallback_task_difficulty="normal",
    )
    assert target_node == "summary_agent"
    assert node_goal == DEFAULT_NODE_GOAL
    assert task_difficulty == "normal"


def test_apply_model_profile_to_routing():
    routing = apply_model_profile_to_routing(
        {"mode": "supervisor_loop"},
        task_difficulty="complex",
        profile={"model": "qwen-max", "think": True},
    )

    assert routing["mode"] == "supervisor_loop"
    assert routing["task_difficulty"] == "complex"
    assert routing["selected_model"] == "qwen-max"
    assert routing["think_enabled"] is True
