from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from app.agent.admin.agent_state import StepFailurePolicy, StepOutput
from app.utils.streaming_utils import invoke, invoke_with_policy


@dataclass(slots=True)
class NodeExecutionResult:
    """
    节点统一执行结果。

    Attributes:
        content: 对用户可见文本。
        status: 步骤状态（completed/failed）。
        error: 失败原因（仅失败时有值）。
        diagnostics: 执行诊断信息（工具调用统计等）。
        stream_chunks: 节点内部流式分片（仅最终节点可能有值）。
    """

    content: str
    status: str
    error: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    stream_chunks: list[str] = field(default_factory=list)


def _resolve_failure_policy(
        policy: StepFailurePolicy | dict[str, Any] | None,
) -> dict[str, Any]:
    """
    解析失败策略并补齐默认值。
    """

    raw = dict(policy or {})

    mode = str(raw.get("mode") or "hybrid").strip().lower()
    if mode not in {"hybrid", "marker_only", "tool_only"}:
        mode = "hybrid"

    error_marker_prefix = str(raw.get("error_marker_prefix") or "__ERROR__:").strip()
    if not error_marker_prefix:
        error_marker_prefix = "__ERROR__:"

    tool_error_counting = str(raw.get("tool_error_counting") or "consecutive").strip().lower()
    if tool_error_counting not in {"consecutive", "total"}:
        tool_error_counting = "consecutive"

    try:
        max_tool_errors = int(raw.get("max_tool_errors") or 2)
    except (TypeError, ValueError):
        max_tool_errors = 2
    if max_tool_errors < 1 or max_tool_errors > 5:
        max_tool_errors = 2

    strict_data_quality = bool(raw.get("strict_data_quality", True))
    return {
        "mode": mode,
        "error_marker_prefix": error_marker_prefix,
        "tool_error_counting": tool_error_counting,
        "max_tool_errors": max_tool_errors,
        "strict_data_quality": strict_data_quality,
    }


def _evaluate_failure_by_policy(
        content: str,
        diagnostics: dict[str, Any] | None,
        policy: StepFailurePolicy | dict[str, Any] | None,
) -> tuple[str, str | None, str]:
    """
    根据策略判定节点执行状态，并返回标准化文本。
    """

    resolved = _resolve_failure_policy(policy)
    mode = str(resolved["mode"])
    marker_prefix = str(resolved["error_marker_prefix"])

    normalized_text = str(content or "").strip()
    marker_hit = bool(marker_prefix) and normalized_text.startswith(marker_prefix)
    if marker_hit:
        marker_reason = normalized_text[len(marker_prefix):].strip() or "模型返回错误标记。"
        normalized_text = marker_reason
        if mode in {"hybrid", "marker_only"}:
            return "failed", marker_reason, normalized_text

    diagnostics = diagnostics or {}
    threshold_hit = bool(diagnostics.get("threshold_hit"))
    if threshold_hit and mode in {"hybrid", "tool_only"}:
        reason = str(diagnostics.get("threshold_reason") or "").strip()
        if not reason:
            reason = "工具失败次数达到阈值。"
        if not normalized_text:
            normalized_text = reason
        return "failed", reason, normalized_text

    return "completed", None, normalized_text


def _build_step_output_update(
        runtime: dict[str, Any],
        *,
        node_name: str,
        status: str,
        text: str = "",
        output: dict[str, Any] | None = None,
        error: str | None = None,
) -> dict[str, Any]:
    """
    基于 runtime.step_id 构建 step_outputs 增量更新。
    """

    step_id = str(runtime.get("step_id") or "").strip()
    if not step_id:
        return {}

    payload: StepOutput = {
        "step_id": step_id,
        "node_name": node_name,
        "status": status,  # type: ignore[typeddict-item]
        "text": text,
        "output": output or {},
    }
    if error:
        payload["error"] = error
    return {"step_outputs": {step_id: payload}}


def _build_failure_policy_kwargs(
        failure_policy: StepFailurePolicy | dict[str, Any] | None,
) -> dict[str, Any]:
    """
    将步骤失败策略转换为 `invoke_with_policy` 调用参数。
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


def execute_tool_node(
        *,
        llm: Any,
        messages: list[Any],
        tools: Sequence[Any],
        enable_stream: bool,
        failure_policy: StepFailurePolicy | dict[str, Any] | None,
        fallback_content: str,
        fallback_error: str,
) -> NodeExecutionResult:
    """
    统一执行“工具型节点”。

    行为：
    - 统一调用失败策略封装；
    - 统一按策略判定 completed/failed；
    - 统一异常兜底输出。
    """

    try:
        content, diagnostics = invoke_with_failure_policy(
            llm=llm,
            messages=messages,
            tools=tools,
            enable_stream=enable_stream,
            failure_policy=failure_policy,
        )
        step_status, failed_error, normalized_content = _evaluate_failure_by_policy(
            content,
            diagnostics,
            failure_policy,
        )
        return NodeExecutionResult(
            content=normalized_content,
            status=step_status,
            error=failed_error,
            diagnostics=diagnostics,
            stream_chunks=list(diagnostics.get("stream_chunks") or []),
        )
    except Exception as exc:
        return NodeExecutionResult(
            content=fallback_content,
            status="failed",
            error=f"{fallback_error}: {exc}",
            diagnostics={},
            stream_chunks=[],
        )


def execute_text_node(
        *,
        llm: Any,
        messages: list[Any],
        fallback_content: str,
        fallback_error: str,
) -> NodeExecutionResult:
    """
    统一执行“纯文本节点”（无工具调用）。
    """

    try:
        content = invoke(llm, messages)
        return NodeExecutionResult(content=content, status="completed")
    except Exception as exc:
        return NodeExecutionResult(
            content=fallback_content,
            status="failed",
            error=f"{fallback_error}: {exc}",
        )


def build_standard_node_update(
        *,
        state: Mapping[str, Any],
        runtime: dict[str, Any],
        node_name: str,
        result_key: str,
        execution_result: NodeExecutionResult,
        is_end: bool | None = None,
        step_output_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    统一构建节点写回结果：
    - `results[result_key]`
    - `step_outputs`（有 step_id 时）
    """

    results = dict(state.get("results") or {})
    result_item: dict[str, Any] = {"content": execution_result.content}
    if is_end is not None:
        result_item["is_end"] = is_end
    if execution_result.stream_chunks:
        result_item["stream_chunks"] = execution_result.stream_chunks
    results[result_key] = result_item

    result: dict[str, Any] = {"results": results}
    output_payload = step_output_payload or {result_key: result_item}
    result.update(
        _build_step_output_update(
            runtime,
            node_name=node_name,
            status=execution_result.status,
            text=execution_result.content,
            output=output_payload,
            error=execution_result.error,
        )
    )
    return result


__all__ = [
    "NodeExecutionResult",
    "build_standard_node_update",
    "execute_text_node",
    "execute_tool_node",
    "invoke_with_failure_policy",
]
