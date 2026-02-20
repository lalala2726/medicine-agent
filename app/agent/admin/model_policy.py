from __future__ import annotations

from typing import Any

# Task difficulty 枚举（英文），非法输入会回退到 NORMAL_DIFFICULTY。
SIMPLE_DIFFICULTY = "simple"
NORMAL_DIFFICULTY = "normal"
COMPLEX_DIFFICULTY = "complex"

ALLOWED_TASK_DIFFICULTIES = {
    SIMPLE_DIFFICULTY,
    NORMAL_DIFFICULTY,
    COMPLEX_DIFFICULTY,
}

# Gateway 路由白名单。
ALLOWED_GATEWAY_ROUTE_TARGETS = {
    "order_agent",
    "product_agent",
    "chat_agent",
    "supervisor_agent",
}

# Supervisor 下一跳白名单。
ALLOWED_SUPERVISOR_NEXT_NODES = {
    "order_agent",
    "product_agent",
    "excel_agent",
    "FINISH",
}


def normalize_task_difficulty(value: Any) -> str:
    """
    将任意输入归一化为受控任务难度枚举。

    Args:
        value: 任意输入值，通常来自模型 JSON 输出或 routing 字段。

    Returns:
        str: 归一化后的任务难度，仅可能是 `simple`、`normal`、`complex`。
            当输入为空或非法时返回 `normal`。
    """

    candidate = str(value or "").strip().lower()
    legacy_map = {
        "简单": SIMPLE_DIFFICULTY,
        "正常": NORMAL_DIFFICULTY,
        "复杂": COMPLEX_DIFFICULTY,
    }
    if candidate in legacy_map:
        return legacy_map[candidate]
    if candidate in ALLOWED_TASK_DIFFICULTIES:
        return candidate
    return NORMAL_DIFFICULTY


def resolve_model_profile(
        task_difficulty: Any,
        *,
        allow_thinking: bool = True,
) -> dict[str, Any]:
    """
    根据任务难度解析模型与深度思考配置。

    固定策略：
    1. `simple` -> `qwen-flash` + `think=False`
    2. `normal` -> `qwen-plus` + `think=False`
    3. `complex` -> `qwen-max` + `think=True`（若 `allow_thinking=False` 则关闭）

    Args:
        task_difficulty: 任务难度，支持任意输入，会先归一化。
        allow_thinking: 是否允许输出 think 开关，默认允许。

    Returns:
        dict[str, Any]: 模型策略字典，字段说明如下：
            - `task_difficulty`: 归一化难度。
            - `model`: 选中的模型名。
            - `think`: 是否开启深度思考。
    """

    normalized = normalize_task_difficulty(task_difficulty)
    if normalized == SIMPLE_DIFFICULTY:
        return {
            "task_difficulty": normalized,
            "model": "qwen-flash",
            "think": False,
        }
    if normalized == NORMAL_DIFFICULTY:
        return {
            "task_difficulty": normalized,
            "model": "qwen-plus",
            "think": False,
        }

    return {
        "task_difficulty": normalized,
        "model": "qwen-max",
        "think": bool(allow_thinking),
    }


def build_gateway_decision(payload: dict[str, Any]) -> tuple[str, str]:
    """
    校验并归一化 Gateway 决策输出。

    Args:
        payload: Gateway LLM 的 JSON 输出对象。

    Returns:
        tuple[str, str]: `(route_target, task_difficulty)`。
            - `route_target` 非法时回退 `supervisor_agent`；
            - `task_difficulty` 非法时回退 `normal`。
    """

    route_target = str(payload.get("route_target") or "").strip()
    if route_target not in ALLOWED_GATEWAY_ROUTE_TARGETS:
        route_target = "supervisor_agent"

    task_difficulty = normalize_task_difficulty(payload.get("task_difficulty"))
    return route_target, task_difficulty


def build_supervisor_decision(
        payload: dict[str, Any],
        *,
        fallback_task_difficulty: Any,
) -> tuple[str, str, str]:
    """
    校验并归一化 Supervisor 决策输出。

    Args:
        payload: Supervisor LLM 的 JSON 输出对象。
        fallback_task_difficulty: 当输出缺失或非法时使用的难度回退值。

    Returns:
        tuple[str, str, str]: `(next_node, directive, task_difficulty)`。
            - `next_node` 非法时回退 `FINISH`；
            - `next_node != FINISH` 且 `directive` 为空时回退 `FINISH`；
            - `FINISH` 场景下 directive 固定为空字符串；
            - 难度字段缺失或非法时回退到 `fallback_task_difficulty`。
    """

    next_node = str(payload.get("next_node") or "").strip()
    if next_node not in ALLOWED_SUPERVISOR_NEXT_NODES:
        next_node = "FINISH"

    directive = str(payload.get("directive") or "").strip()
    if next_node != "FINISH" and not directive:
        next_node = "FINISH"
        directive = ""
    if next_node == "FINISH":
        directive = ""

    if "task_difficulty" in payload:
        task_difficulty = normalize_task_difficulty(payload.get("task_difficulty"))
    else:
        task_difficulty = normalize_task_difficulty(fallback_task_difficulty)

    return next_node, directive, task_difficulty


def apply_model_profile_to_routing(
        routing: dict[str, Any],
        *,
        task_difficulty: Any,
        profile: dict[str, Any],
) -> dict[str, Any]:
    """
    将难度与模型策略写回 routing 元信息。

    Args:
        routing: 现有 routing 字典。
        task_difficulty: 当前任务难度值。
        profile: 由 `resolve_model_profile` 生成的模型策略字典。

    Returns:
        dict[str, Any]: 新 routing 字典，至少包含：
            - `task_difficulty`: 归一化任务难度；
            - `selected_model`: 当前建议模型；
            - `think_enabled`: 当前是否开启深度思考。
    """

    merged = dict(routing or {})
    normalized = normalize_task_difficulty(task_difficulty)
    merged["task_difficulty"] = normalized
    merged["selected_model"] = str(profile.get("model") or "qwen-plus")
    merged["think_enabled"] = bool(profile.get("think"))
    return merged


__all__ = [
    "SIMPLE_DIFFICULTY",
    "NORMAL_DIFFICULTY",
    "COMPLEX_DIFFICULTY",
    "ALLOWED_TASK_DIFFICULTIES",
    "ALLOWED_GATEWAY_ROUTE_TARGETS",
    "ALLOWED_SUPERVISOR_NEXT_NODES",
    "normalize_task_difficulty",
    "resolve_model_profile",
    "build_gateway_decision",
    "build_supervisor_decision",
    "apply_model_profile_to_routing",
]
