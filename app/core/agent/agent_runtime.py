from __future__ import annotations

import asyncio
import contextvars
import json
import os
from typing import Any, Mapping, Sequence

from langchain_core.messages import ToolMessage
from loguru import logger

from app.core.agent.agent_event_bus import emit_function_call, emit_sse_response
from app.core.agent.agent_tool_events import resolve_tool_call_messages
from app.core.agent.agent_tool_trace import (
    _aggregate_usage_from_ai_messages,
    _build_tool_call_traces,
    _is_ai_message,
    _resolve_model_name_from_response,
    _split_generated_messages,
    extract_text,
    extract_usage_from_response,
)
from app.schemas.sse_response import AssistantResponse, Content, MessageType

_TOOL_TRACE_STACK: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar(
    "admin_tool_trace_stack",
    default=None,
)


def _is_tool_log_enabled() -> bool:
    """
    检查是否启用工具调用日志。

    Returns:
        bool: 环境变量 `AGENT_TOOL_LOG_ENABLED` 启用时返回 True。
    """

    value = os.getenv("AGENT_TOOL_LOG_ENABLED", "false").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _empty_runtime_diagnostics() -> dict[str, Any]:
    """
    返回运行时追踪使用的空诊断结构。

    Returns:
        dict[str, Any]: 统一的空诊断结构。
    """

    return {
        "tool_call_details": [],
        "llm_usage": None,
        "llm_usage_complete": True,
        "model_name": "unknown",
    }


def _mark_current_tool_llm_trace(
    *,
    usage: dict[str, int] | None,
    is_usage_complete: bool,
) -> None:
    """
    将当前工具内部 LLM 调用结果写入工具追踪栈顶。

    Args:
        usage: 当前调用 usage。
        is_usage_complete: usage 完整性标记。
    """

    current_stack = _TOOL_TRACE_STACK.get()
    if not current_stack:
        return
    current_tool = current_stack[-1]
    current_tool["llm_used"] = True
    current_tool["llm_usage_complete"] = bool(is_usage_complete)
    if usage is not None:
        current_tool["llm_token_usage"] = dict(usage)


def run_model_with_trace(
    llm: Any,
    messages: list[Any],
    *,
    tools: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """
    统一执行模型并返回文本 + usage + 工具调用轨迹。

    Args:
        llm: LangChain ChatModel 实例。
        messages: 消息列表。
        tools: 可选工具集合。

    Returns:
        dict[str, Any]:
            - text: 模型输出文本；
            - model_name: 模型名称；
            - usage: token 使用量；
            - is_usage_complete: usage 完整性；
            - tool_calls: 工具调用追踪；
            - raw_content: 原始内容。
    """

    if tools:
        content, diagnostics = _invoke_with_tools_with_diagnostics(
            llm,
            messages,
            tools,
        )
        usage_payload = diagnostics.get("llm_usage")
        usage = usage_payload if isinstance(usage_payload, dict) else None
        is_usage_complete = bool(diagnostics.get("llm_usage_complete", usage is not None))
        _mark_current_tool_llm_trace(
            usage=usage,
            is_usage_complete=is_usage_complete,
        )
        return {
            "text": content,
            "model_name": str(diagnostics.get("model_name") or "unknown"),
            "usage": usage,
            "is_usage_complete": is_usage_complete,
            "tool_calls": list(diagnostics.get("tool_call_details") or []),
            "raw_content": content,
        }

    response = llm.invoke(messages)
    text = extract_text(response)
    usage = extract_usage_from_response(response)
    is_usage_complete = usage is not None
    _mark_current_tool_llm_trace(
        usage=usage,
        is_usage_complete=is_usage_complete,
    )
    return {
        "text": text,
        "model_name": _resolve_model_name_from_response(
            response,
            fallback=str(getattr(llm, "model_name", "unknown") or "unknown"),
        ),
        "usage": usage,
        "is_usage_complete": is_usage_complete,
        "tool_calls": [],
        "raw_content": getattr(response, "content", None),
    }


def run_agent_with_trace(
    agent_instance: Any,
    messages: list[Any],
) -> dict[str, Any]:
    """
    执行 `create_agent_instance` 产物，并返回统一追踪结构。

    Args:
        agent_instance: `create_agent_instance` 创建的 agent 实例。
        messages: 输入消息列表。

    Returns:
        dict[str, Any]: 与 `run_model_with_trace` 一致的追踪结构。
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

    raw_final_messages = latest_state.get("messages")
    final_messages = list(raw_final_messages) if isinstance(raw_final_messages, list) else []
    generated_messages = _split_generated_messages(final_messages, messages)

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
        text = "".join(streamed_chunks)

    usage, is_usage_complete = _aggregate_usage_from_ai_messages(preferred_messages)
    tool_calls = _build_tool_call_traces(preferred_messages)
    model_name = (
        _resolve_model_name_from_response(last_ai_message)
        if last_ai_message is not None
        else "unknown"
    )
    raw_content = getattr(last_ai_message, "content", None) if last_ai_message is not None else text

    _mark_current_tool_llm_trace(
        usage=usage,
        is_usage_complete=is_usage_complete,
    )

    return {
        "text": text,
        "model_name": model_name,
        "usage": usage,
        "is_usage_complete": is_usage_complete,
        "tool_calls": tool_calls,
        "raw_content": raw_content,
    }


def _run_async(coro: Any) -> Any:
    """
    在当前线程运行异步协程。

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


def _exec_tool_with_meta(
    tool_call: dict,
    tool_map: dict[str, Any],
    log_enabled: bool | None = None,
) -> tuple[str, bool, str, dict[str, Any]]:
    """
    执行单个工具调用并返回元信息。

    Args:
        tool_call: 工具调用数据。
        tool_map: 工具名称到工具对象的映射。
        log_enabled: 是否启用工具日志。

    Returns:
        tuple[str, bool, str, dict[str, Any]]:
            - result_text: 工具返回文本；
            - is_error: 是否错误；
            - error_message: 错误文案；
            - tool_detail: 追踪明细。
    """

    name = tool_call["name"]
    args = tool_call.get("args", {})
    tool_fn = tool_map.get(name)
    tool_node = f"tool:{name}"
    parent_stack = _TOOL_TRACE_STACK.get() or []
    tool_detail: dict[str, Any] = {
        "tool_name": name,
        "tool_input": args,
        "is_error": False,
        "error_message": None,
        "llm_used": False,
        "llm_usage_complete": True,
        "llm_token_usage": None,
        "children": [],
    }
    if parent_stack:
        parent_stack[-1].setdefault("children", []).append(tool_detail)
    stack_token = _TOOL_TRACE_STACK.set([*parent_stack, tool_detail])

    try:
        if log_enabled is None:
            log_enabled = _is_tool_log_enabled()

        if tool_fn is None:
            start_message, _, _ = resolve_tool_call_messages(name)
            unknown_message = f"未知工具: {name}"
            emit_function_call(
                node=tool_node,
                state="start",
                message=start_message,
            )
            emit_function_call(
                node=tool_node,
                state="end",
                result="error",
                message=unknown_message,
            )
            if log_enabled:
                logger.warning("未知工具: {}", name)
            tool_detail["is_error"] = True
            tool_detail["error_message"] = unknown_message
            return (
                json.dumps({"error": unknown_message}, ensure_ascii=False),
                True,
                unknown_message,
                tool_detail,
            )

        if log_enabled:
            logger.info("工具调用: name={} args={}", name, args)

        start_message, error_tip_message, _ = resolve_tool_call_messages(name)
        emit_function_call(
            node=tool_node,
            state="start",
            message=start_message,
            name=name,
            arguments=(
                json.dumps(args, ensure_ascii=False, default=str)
                if isinstance(args, (dict, list))
                else str(args)
            ),
        )

        result = _run_async(tool_fn.ainvoke(args))
        if log_enabled:
            logger.info(
                "工具返回: name={} result={}",
                name,
                json.dumps(result, ensure_ascii=False, default=str)[:500],
            )
        result_text = (
            json.dumps(result, ensure_ascii=False, default=str)
            if isinstance(result, (dict, list))
            else str(result)
        )
        is_error = isinstance(result, dict) and "error" in result
        error_message = str(result.get("error") or "").strip() if isinstance(result, dict) else ""
        tool_detail["is_error"] = is_error
        tool_detail["error_message"] = error_message or None
        emit_function_call(
            node=tool_node,
            state="end",
            result="error" if is_error else "success",
            message=error_message or ("工具调用成功" if not is_error else error_tip_message),
            name=name,
        )
        return (
            result_text,
            is_error,
            error_message,
            tool_detail,
        )
    except Exception as exc:
        if log_enabled:
            logger.error("工具执行失败: name={} error={}", name, exc)
        message = f"工具执行失败: {name}, {exc}"
        tool_detail["is_error"] = True
        tool_detail["error_message"] = message
        emit_function_call(
            node=tool_node,
            state="end",
            result="error",
            message=message,
            name=name,
        )
        return (
            json.dumps({"error": message}, ensure_ascii=False),
            True,
            message,
            tool_detail,
        )
    finally:
        _TOOL_TRACE_STACK.reset(stack_token)


def _invoke_with_tools_with_diagnostics(
    llm: Any,
    messages: list[Any],
    tools: Sequence[Any],
) -> tuple[str, dict[str, Any]]:
    """
    带工具诊断信息的同步 Agent 执行。

    Args:
        llm: 模型实例。
        messages: 消息列表。
        tools: 工具列表。

    Returns:
        tuple[str, dict[str, Any]]: `(content, diagnostics)`。
    """

    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}
    log_enabled = _is_tool_log_enabled()

    diagnostics = _empty_runtime_diagnostics()
    usage_prompt = 0
    usage_completion = 0
    usage_total = 0
    usage_complete = True

    while True:
        response = llm_with_tools.invoke(messages)
        model_name = _resolve_model_name_from_response(
            response,
            fallback=str(getattr(llm, "model_name", "unknown") or "unknown"),
        )
        diagnostics["model_name"] = model_name
        response_usage = extract_usage_from_response(response)
        if response_usage is None:
            usage_complete = False
        else:
            usage_prompt += response_usage["prompt_tokens"]
            usage_completion += response_usage["completion_tokens"]
            usage_total += response_usage["total_tokens"]
        messages.append(response)

        tool_calls = getattr(response, "tool_calls", None)
        if not tool_calls:
            diagnostics["llm_usage"] = {
                "prompt_tokens": usage_prompt,
                "completion_tokens": usage_completion,
                "total_tokens": usage_total,
            }
            diagnostics["llm_usage_complete"] = usage_complete
            return extract_text(response), diagnostics

        if log_enabled:
            logger.info("触发 {} 个工具调用", len(tool_calls))

        for tc in tool_calls:
            result, _, _, tool_detail = _exec_tool_with_meta(tc, tool_map, log_enabled)
            diagnostics["tool_call_details"].append(tool_detail)
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
