from __future__ import annotations

import asyncio
import contextvars
from typing import Any, Mapping

from loguru import logger

from app.core.agent.agent_event_bus import emit_sse_response
from app.core.agent.agent_tool_trace import (
    _aggregate_usage_from_ai_messages,
    _build_tool_call_traces,
    _is_ai_message,
    _resolve_model_name_from_response,
    _split_generated_messages,
    extract_text,
)
from app.schemas.sse_response import AssistantResponse, Content, MessageType


def _run_async(coro: Any) -> Any:
    """
    在同步上下文安全执行异步协程。

    Args:
        coro: 协程对象。

    Returns:
        Any: 协程返回值。
    """

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        current_context = contextvars.copy_context()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(current_context.run, asyncio.run, coro).result()

    return asyncio.run(coro)


def _invoke_agent(agent_instance: Any, payload: dict[str, Any]) -> Any:
    """
    执行 agent 调用，优先走异步入口以兼容 async tools。

    Args:
        agent_instance: `create_agent_instance` 返回的 agent。
        payload: 调用入参。

    Returns:
        Any: agent 返回结果。
    """

    ainvoke = getattr(agent_instance, "ainvoke", None)
    if callable(ainvoke):
        return _run_async(ainvoke(payload))
    return agent_instance.invoke(payload)


def _resolve_final_messages(payload: Any) -> list[Any]:
    """
    从 agent 执行结果中提取最终消息列表。

    Args:
        payload: `agent.invoke(...)` 或 `stream_mode="values"` 的状态载荷。

    Returns:
        list[Any]: 最终消息列表；未命中时返回空列表。
    """

    if not isinstance(payload, Mapping):
        return []

    raw_messages = payload.get("messages")
    if isinstance(raw_messages, list):
        return list(raw_messages)
    return []


def _build_trace_result(
    *,
    final_messages: list[Any],
    input_messages: list[Any],
    fallback_text: str,
) -> dict[str, Any]:
    """
    根据最终消息构建统一追踪结果。

    Args:
        final_messages: agent 执行完成后的完整消息序列。
        input_messages: 本次调用输入消息序列。
        fallback_text: 在无法从最终 AI 消息提取文本时的兜底文本。

    Returns:
        dict[str, Any]: 统一追踪结构。
    """

    generated_messages = _split_generated_messages(final_messages, input_messages)
    preferred_messages = generated_messages or final_messages

    last_ai_message = next(
        (message for message in reversed(preferred_messages) if _is_ai_message(message)),
        None,
    )
    if last_ai_message is None:
        last_ai_message = next(
            (message for message in reversed(final_messages) if _is_ai_message(message)),
            None,
        )

    text = extract_text(last_ai_message) if last_ai_message is not None else ""
    if not text:
        text = fallback_text

    usage, is_usage_complete = _aggregate_usage_from_ai_messages(preferred_messages)
    tool_calls = _build_tool_call_traces(preferred_messages)
    model_name = (
        _resolve_model_name_from_response(last_ai_message)
        if last_ai_message is not None
        else "unknown"
    )
    raw_content = (
        getattr(last_ai_message, "content", None)
        if last_ai_message is not None
        else text
    )

    return {
        "text": text,
        "model_name": model_name,
        "usage": usage,
        "is_usage_complete": is_usage_complete,
        "tool_calls": tool_calls,
        "raw_content": raw_content,
    }


def _log_tool_errors(tool_calls: list[dict[str, Any]]) -> None:
    """
    打印工具调用错误摘要日志。

    Args:
        tool_calls: 追踪结构中的工具调用列表。

    Returns:
        None
    """

    error_items = [
        {
            "tool_name": str(item.get("tool_name") or ""),
            "error_message": str(item.get("error_message") or ""),
            "tool_input": item.get("tool_input"),
        }
        for item in tool_calls
        if isinstance(item, dict) and bool(item.get("is_error"))
    ]
    if not error_items:
        return
    logger.warning(
        "Agent tool calls contain errors. error_count={} errors={}",
        len(error_items),
        error_items,
    )


def run_agent_with_trace(
    agent_instance: Any,
    messages: list[Any],
) -> dict[str, Any]:
    """
    以流式模式执行 agent，并输出统一追踪结果。

    该入口用于需要在运行过程中主动发送 ANSWER 分片的节点（如 supervisor）。

    Args:
        agent_instance: `create_agent_instance` 返回的可执行 agent。
        messages: 输入消息列表。

    Returns:
        dict[str, Any]: 统一追踪结构（text/model_name/usage/is_usage_complete/tool_calls/raw_content）。
    """

    streamed_chunks: list[str] = []
    latest_state: dict[str, Any] = {}

    for raw_event in agent_instance.stream(
        {"messages": messages},
        stream_mode=["messages", "values"],
    ):
        if not isinstance(raw_event, tuple) or len(raw_event) != 2:
            continue
        mode, payload = raw_event

        if mode == "values":
            if isinstance(payload, Mapping):
                latest_state = dict(payload)
            continue

        if mode != "messages":
            continue
        if not isinstance(payload, tuple) or len(payload) != 2:
            continue

        message_chunk, metadata = payload
        if not isinstance(metadata, Mapping):
            continue
        if str(metadata.get("langgraph_node") or "") != "model":
            continue

        delta = extract_text(message_chunk)
        if not delta:
            continue

        streamed_chunks.append(delta)
        emit_sse_response(
            AssistantResponse(
                type=MessageType.ANSWER,
                content=Content(text=delta),
            )
        )

    final_messages = _resolve_final_messages(latest_state)
    trace_result = _build_trace_result(
        final_messages=final_messages,
        input_messages=messages,
        fallback_text="".join(streamed_chunks),
    )
    _log_tool_errors(list(trace_result.get("tool_calls") or []))
    return trace_result


def run_agent_invoke_with_trace(
    agent_instance: Any,
    messages: list[Any],
) -> dict[str, Any]:
    """
    以非流式 invoke 模式执行 agent，并输出统一追踪结果。

    Args:
        agent_instance: `create_agent_instance` 返回的可执行 agent。
        messages: 输入消息列表。

    Returns:
        dict[str, Any]: 统一追踪结构（text/model_name/usage/is_usage_complete/tool_calls/raw_content）。
    """

    result = _invoke_agent(agent_instance, {"messages": messages})
    final_messages = _resolve_final_messages(result)

    fallback_text = ""
    if isinstance(result, Mapping):
        fallback_text = str(result.get("output") or result.get("text") or "")

    trace_result = _build_trace_result(
        final_messages=final_messages,
        input_messages=messages,
        fallback_text=fallback_text,
    )
    _log_tool_errors(list(trace_result.get("tool_calls") or []))
    return trace_result
