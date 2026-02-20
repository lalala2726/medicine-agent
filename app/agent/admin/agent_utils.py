from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from app.utils.streaming_utils import invoke_with_policy

UNKNOWN_MODEL_NAME = "unknown"


@dataclass(slots=True)
class NodeExecutionResult:
    """
    Standardized worker-node execution result.
    """

    content: str
    status: str
    error: str | None = None
    model_name: str = UNKNOWN_MODEL_NAME
    input_messages: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    stream_chunks: list[str] = field(default_factory=list)


def _resolve_llm_model_name(llm: Any) -> str:
    for attr in ("model_name", "model"):
        candidate = getattr(llm, attr, None)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return UNKNOWN_MODEL_NAME


def _serialize_message(message: Any) -> dict[str, Any]:
    role = str(getattr(message, "type", "") or message.__class__.__name__).strip().lower()
    content = getattr(message, "content", "")
    message_dict: dict[str, Any] = {
        "role": role or "unknown",
        "content": content if content is not None else "",
    }
    additional_kwargs = getattr(message, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict) and additional_kwargs:
        message_dict["additional_kwargs"] = additional_kwargs
    name = getattr(message, "name", None)
    if isinstance(name, str) and name:
        message_dict["name"] = name
    return message_dict


def _serialize_messages(messages: list[Any]) -> list[dict[str, Any]]:
    return [_serialize_message(item) for item in messages]


def build_execution_trace_update(
        *,
        node_name: str,
        model_name: str = UNKNOWN_MODEL_NAME,
        input_messages: list[Any] | None = None,
        output_text: str = "",
        tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Build `execution_traces` incremental update payload.
    """

    trace_item = {
        "node_name": str(node_name or UNKNOWN_MODEL_NAME),
        "model_name": str(model_name or UNKNOWN_MODEL_NAME),
        "input_messages": list(input_messages or []),
        "output_text": str(output_text or ""),
        "tool_calls": list(tool_calls or []),
    }
    return {"execution_traces": [trace_item]}


def _resolve_invoke_policy(failure_policy: dict[str, Any] | None) -> dict[str, Any]:
    policy = dict(failure_policy or {})
    return {
        "error_marker_prefix": str(policy.get("error_marker_prefix") or "__ERROR__:"),
        "tool_error_counting": str(policy.get("tool_error_counting") or "consecutive"),
        "max_tool_errors": int(policy.get("max_tool_errors") or 2),
    }


def _resolve_result_status(
        content: str,
        diagnostics: dict[str, Any] | None,
        *,
        error_marker_prefix: str,
) -> tuple[str, str | None, str]:
    normalized_content = str(content or "").strip()
    if normalized_content.startswith(error_marker_prefix):
        marker_reason = normalized_content[len(error_marker_prefix):].strip() or "模型返回错误标记。"
        return "failed", marker_reason, marker_reason

    diagnostics = diagnostics or {}
    if bool(diagnostics.get("threshold_hit")):
        reason = str(diagnostics.get("threshold_reason") or "").strip() or "工具失败次数达到阈值。"
        if not normalized_content:
            normalized_content = reason
        return "failed", reason, normalized_content
    return "completed", None, normalized_content


def execute_tool_node(
        *,
        llm: Any,
        messages: list[Any],
        tools: Sequence[Any],
        enable_stream: bool,
        failure_policy: dict[str, Any] | None,
        fallback_content: str,
        fallback_error: str,
) -> NodeExecutionResult:
    """
    Execute a tool-enabled worker node under unified error handling.
    """

    model_name = _resolve_llm_model_name(llm)
    serialized_inputs = _serialize_messages(messages)
    invoke_policy = _resolve_invoke_policy(failure_policy)
    marker_prefix = str(invoke_policy["error_marker_prefix"])

    try:
        content, diagnostics = invoke_with_policy(
            llm,
            messages,
            tools=tools,
            enable_stream=enable_stream,
            error_marker_prefix=marker_prefix,
            tool_error_counting=str(invoke_policy["tool_error_counting"]),
            max_tool_errors=int(invoke_policy["max_tool_errors"]),
        )
        step_status, failed_error, normalized_content = _resolve_result_status(
            content,
            diagnostics,
            error_marker_prefix=marker_prefix,
        )
        return NodeExecutionResult(
            content=normalized_content,
            status=step_status,
            error=failed_error,
            model_name=model_name,
            input_messages=serialized_inputs,
            tool_calls=list((diagnostics or {}).get("tool_call_details") or []),
            diagnostics=diagnostics or {},
            stream_chunks=list((diagnostics or {}).get("stream_chunks") or []),
        )
    except Exception as exc:
        return NodeExecutionResult(
            content=fallback_content,
            status="failed",
            error=f"{fallback_error}: {exc}",
            model_name=model_name,
            input_messages=serialized_inputs,
            tool_calls=[],
            diagnostics={},
            stream_chunks=[],
        )


__all__ = [
    "UNKNOWN_MODEL_NAME",
    "NodeExecutionResult",
    "build_execution_trace_update",
    "execute_tool_node",
]
