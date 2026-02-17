from __future__ import annotations

from typing import Any, Sequence

from app.agent.admin.agent_state import StepFailurePolicy
from app.utils.streaming_utils import invoke_with_policy


def _build_failure_policy_kwargs(
        failure_policy: StepFailurePolicy | dict[str, Any] | None,
) -> dict[str, Any]:
    """
    将步骤失败策略转换为 `invoke_with_policy` 调用参数。

    统一在这里收敛默认值，避免各节点重复拼装：
    - error_marker_prefix 默认 `__ERROR__:`
    - tool_error_counting 默认 `consecutive`
    - max_tool_errors 默认 `2`
    """

    policy = failure_policy or {}
    return {
        "error_marker_prefix": str(policy.get("error_marker_prefix") or "__ERROR__:"),
        "tool_error_counting": str(policy.get("tool_error_counting") or "consecutive"),
        "max_tool_errors": int(policy.get("max_tool_errors") or 2),
    }


def invoke_with_failure_policy(
        *,
        llm: Any,
        messages: list[Any],
        tools: Sequence[Any] | None,
        enable_stream: bool,
        failure_policy: StepFailurePolicy | dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    """
    按统一失败策略调用模型/工具执行。

    该函数封装了所有与 failure_policy 相关的参数映射逻辑，
    节点侧只需要关心业务输入，不再重复写默认值与字段读取。
    """

    invoke_kwargs = _build_failure_policy_kwargs(failure_policy)
    return invoke_with_policy(
        llm,
        messages,
        tools=tools,
        enable_stream=enable_stream,
        error_marker_prefix=invoke_kwargs["error_marker_prefix"],
        tool_error_counting=invoke_kwargs["tool_error_counting"],
        max_tool_errors=invoke_kwargs["max_tool_errors"],
    )


__all__ = ["invoke_with_failure_policy"]
