from __future__ import annotations

import asyncio
import contextvars
import json
import os
from typing import Any, Mapping, Optional, Sequence

from langchain_core.messages import ToolMessage
from loguru import logger

from app.core.agent.agent_event_bus import emit_function_call, emit_sse_response
from app.core.agent.agent_tool_events import resolve_tool_call_messages
from app.schemas.sse_response import AssistantResponse, Content, MessageType

MAX_TOOL_ROUNDS = 20

_TOOL_TRACE_STACK: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar(
    "admin_tool_trace_stack",
    default=None,
)


def _empty_policy_diagnostics() -> dict[str, Any]:
    """
    返回失败策略判定使用的空诊断结构。

    Args:
        无。

    Returns:
        dict[str, Any]: 统一的空诊断结构。
    """

    return {
        "tool_calls": 0,
        "tool_errors_total": 0,
        "tool_errors_consecutive_peak": 0,
        "threshold_hit": False,
        "threshold_reason": "",
        "tool_error_messages": [],
        "tool_call_details": [],
        "stream_chunks": [],
        "reasoning_chunks": [],
        "llm_usage": None,
        "llm_usage_complete": True,
        "model_name": "unknown",
    }


def _is_tool_log_enabled() -> bool:
    """
    检查是否启用工具调用日志。

    Args:
        无。

    Returns:
        bool: 环境变量 `AGENT_TOOL_LOG_ENABLED` 启用时返回 True。
    """

    value = os.getenv("AGENT_TOOL_LOG_ENABLED", "false").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _to_non_negative_int(value: Any) -> int | None:
    """
    将值转换为非负整数。

    Args:
        value: 任意待转换对象。

    Returns:
        int | None: 转换成功返回非负整数，失败返回 None。
    """

    if value is None:
        return None
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return None
    if resolved < 0:
        return None
    return resolved


def normalize_usage_payload(raw: Mapping[str, Any] | None) -> dict[str, int] | None:
    """
    标准化 provider usage 结构。

    兼容字段：
    - prompt_tokens / completion_tokens / total_tokens
    - input_tokens / output_tokens / total_tokens

    Args:
        raw: 原始 usage 字典。

    Returns:
        dict[str, int] | None: 标准化后的 usage，无有效字段时返回 None。
    """

    if raw is None:
        return None

    prompt_tokens = _to_non_negative_int(raw.get("prompt_tokens"))
    if prompt_tokens is None:
        prompt_tokens = _to_non_negative_int(raw.get("input_tokens"))

    completion_tokens = _to_non_negative_int(raw.get("completion_tokens"))
    if completion_tokens is None:
        completion_tokens = _to_non_negative_int(raw.get("output_tokens"))

    total_tokens = _to_non_negative_int(raw.get("total_tokens"))
    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return None

    resolved_prompt = prompt_tokens or 0
    resolved_completion = completion_tokens or 0
    resolved_total = total_tokens
    if resolved_total is None:
        resolved_total = resolved_prompt + resolved_completion

    return {
        "prompt_tokens": resolved_prompt,
        "completion_tokens": resolved_completion,
        "total_tokens": resolved_total,
    }


def extract_usage_from_response(response: Any) -> dict[str, int] | None:
    """
    从 LLM 响应中提取 usage。

    Args:
        response: 模型响应对象。

    Returns:
        dict[str, int] | None: usage 字典，未命中时返回 None。
    """

    usage_metadata = getattr(response, "usage_metadata", None)
    if isinstance(usage_metadata, Mapping):
        normalized = normalize_usage_payload(usage_metadata)
        if normalized is not None:
            return normalized

    response_metadata = getattr(response, "response_metadata", None)
    if isinstance(response_metadata, Mapping):
        for key in ("token_usage", "usage", "usage_metadata"):
            candidate = response_metadata.get(key)
            if isinstance(candidate, Mapping):
                normalized = normalize_usage_payload(candidate)
                if normalized is not None:
                    return normalized
    return None


def _resolve_model_name_from_response(response: Any, fallback: str = "unknown") -> str:
    """
    从响应元数据解析模型名。

    Args:
        response: 模型响应对象。
        fallback: 兜底模型名。

    Returns:
        str: 解析到的模型名或 fallback。
    """

    response_metadata = getattr(response, "response_metadata", None)
    if isinstance(response_metadata, Mapping):
        for key in ("model_name", "model", "model_id"):
            candidate = response_metadata.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return fallback


def extract_text(message: Any) -> str:
    """
    从消息对象中提取纯文本内容。

    Args:
        message: LangChain 消息对象（AIMessage/Chunk 等）。

    Returns:
        str: 纯文本内容，无内容时返回空字符串。
    """

    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return "" if content is None else str(content)


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

    Returns:
        None: 更新上下文追踪栈，不返回值。
    """

    current_stack = _TOOL_TRACE_STACK.get()
    if not current_stack:
        return
    current_tool = current_stack[-1]
    current_tool["llm_used"] = True
    current_tool["llm_usage_complete"] = bool(is_usage_complete)
    if usage is not None:
        current_tool["llm_token_usage"] = dict(usage)


def _normalize_message_signature(message: Any) -> tuple[str, str]:
    """
    提取消息类型与文本，用于判断输入前缀。

    Args:
        message: 任意消息对象或字典。

    Returns:
        tuple[str, str]: `(message_type, text_content)`。
    """

    if isinstance(message, Mapping):
        role = str(message.get("type") or message.get("role") or "unknown").lower()
        raw_content = message.get("content")
        if isinstance(raw_content, str):
            return role, raw_content
        if isinstance(raw_content, list):
            parts: list[str] = []
            for item in raw_content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, Mapping):
                    parts.append(str(item.get("text") or ""))
            return role, "".join(parts)
        return role, "" if raw_content is None else str(raw_content)

    role = str(getattr(message, "type", message.__class__.__name__) or "unknown").lower()
    return role, extract_text(message)


def _split_generated_messages(
        final_messages: list[Any],
        input_messages: list[Any],
) -> list[Any]:
    """
    尝试剥离输入前缀，只保留本次 Agent 新增消息。

    Args:
        final_messages: agent 最终状态消息。
        input_messages: 本轮输入消息。

    Returns:
        list[Any]: 剥离后的新增消息列表。
    """

    if len(final_messages) < len(input_messages):
        return list(final_messages)
    for index, input_message in enumerate(input_messages):
        if _normalize_message_signature(final_messages[index]) != _normalize_message_signature(input_message):
            return list(final_messages)
    return list(final_messages[len(input_messages):])


def _is_ai_message(message: Any) -> bool:
    """
    判断消息是否为 AI 消息。

    Args:
        message: 任意消息对象。

    Returns:
        bool: 是 AI 消息返回 True，否则 False。
    """

    return str(getattr(message, "type", "") or "").lower() == "ai"


def _aggregate_usage_from_ai_messages(
        messages: list[Any],
) -> tuple[dict[str, int] | None, bool]:
    """
    汇总 AI 消息 usage，并返回完整性标记。

    Args:
        messages: 消息列表。

    Returns:
        tuple[dict[str, int] | None, bool]:
            - usage 汇总；
            - usage 是否完整。
    """

    ai_messages = [message for message in messages if _is_ai_message(message)]
    if not ai_messages:
        return None, False

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    has_usage = False
    is_complete = True

    for message in ai_messages:
        usage = extract_usage_from_response(message)
        if usage is None:
            is_complete = False
            continue
        has_usage = True
        prompt_tokens += usage["prompt_tokens"]
        completion_tokens += usage["completion_tokens"]
        total_tokens += usage["total_tokens"]

    if not has_usage:
        return None, False
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }, is_complete


def _extract_tool_error_from_content(content: Any) -> tuple[bool, str | None]:
    """
    从 ToolMessage 内容推断是否报错。

    Args:
        content: ToolMessage 的 content 字段。

    Returns:
        tuple[bool, str | None]: `(is_error, error_message)`。
    """

    if isinstance(content, Mapping):
        error_value = content.get("error")
        if error_value:
            return True, str(error_value)
        return False, None

    if isinstance(content, list):
        for item in content:
            is_error, error_message = _extract_tool_error_from_content(item)
            if is_error:
                return True, error_message
        return False, None

    text = str(content or "").strip()
    if not text:
        return False, None

    if "__ERROR__:" in text:
        error_message = text.split("__ERROR__:", 1)[1].strip()
        return True, error_message or text

    try:
        parsed = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False, None

    if isinstance(parsed, Mapping):
        error_value = parsed.get("error")
        if error_value:
            return True, str(error_value)
    return False, None


def _build_tool_call_traces(messages: list[Any]) -> list[dict[str, Any]]:
    """
    从最终消息序列构建工具调用追踪结构。

    Args:
        messages: 最终消息列表。

    Returns:
        list[dict[str, Any]]: 结构化工具调用追踪明细。
    """

    tool_messages: dict[str, ToolMessage] = {}
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        tool_call_id = str(getattr(message, "tool_call_id", "") or "").strip()
        if not tool_call_id:
            continue
        tool_messages[tool_call_id] = message

    tool_calls: list[dict[str, Any]] = []
    for message in messages:
        if not _is_ai_message(message):
            continue
        raw_tool_calls = getattr(message, "tool_calls", None)
        if not isinstance(raw_tool_calls, list):
            continue
        for raw_call in raw_tool_calls:
            if not isinstance(raw_call, Mapping):
                continue
            tool_call_id = str(raw_call.get("id") or "").strip()
            matched_tool_message = tool_messages.get(tool_call_id)
            is_error, error_message = _extract_tool_error_from_content(
                getattr(matched_tool_message, "content", None)
            )
            tool_calls.append(
                {
                    "tool_name": str(raw_call.get("name") or ""),
                    "tool_input": raw_call.get("args"),
                    "is_error": is_error,
                    "error_message": error_message,
                    "llm_used": False,
                    "llm_usage_complete": True,
                    "llm_token_usage": None,
                    "children": [],
                }
            )
    return tool_calls


def run_model_with_trace(
        llm: Any,
        messages: list[Any],
        *,
        tools: Optional[Sequence[Any]] = None,
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
) -> dict[str, Any]:
    """
    统一执行模型并返回文本 + usage + 工具调用轨迹。

    Args:
        llm: LangChain ChatModel 实例。
        messages: 消息列表。
        tools: 可选工具集合。
        max_tool_rounds: 工具调用最大轮次。

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
            max_tool_rounds,
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
        max_rounds: int,
        *,
        error_marker_prefix: str = "__ERROR__:",
        tool_error_counting: str = "consecutive",
        max_tool_errors: int = 2,
) -> tuple[str, dict[str, Any]]:
    """
    带工具诊断信息的同步 Agent 执行。

    Args:
        llm: 模型实例。
        messages: 消息列表。
        tools: 工具列表。
        max_rounds: 最大工具轮次。
        error_marker_prefix: 错误文本前缀。
        tool_error_counting: 失败计数模式。
        max_tool_errors: 最大允许工具失败次数。

    Returns:
        tuple[str, dict[str, Any]]: `(content, diagnostics)`。
    """

    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}
    log_enabled = _is_tool_log_enabled()

    diagnostics = _empty_policy_diagnostics()
    tool_errors_consecutive = 0
    usage_prompt = 0
    usage_completion = 0
    usage_total = 0
    usage_complete = True

    def _threshold_reached() -> bool:
        if tool_error_counting == "total":
            return int(diagnostics["tool_errors_total"]) >= max_tool_errors
        return tool_errors_consecutive >= max_tool_errors

    for round_idx in range(max_rounds):
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
            if log_enabled:
                logger.info("第 {} 轮无工具调用，返回最终结果", round_idx + 1)
            diagnostics["llm_usage"] = {
                "prompt_tokens": usage_prompt,
                "completion_tokens": usage_completion,
                "total_tokens": usage_total,
            }
            diagnostics["llm_usage_complete"] = usage_complete
            return extract_text(response), diagnostics

        if log_enabled:
            logger.info(
                "第 {} 轮触发 {} 个工具调用", round_idx + 1, len(tool_calls)
            )

        for tc in tool_calls:
            diagnostics["tool_calls"] = int(diagnostics["tool_calls"]) + 1
            result, is_error, error_message, tool_detail = _exec_tool_with_meta(tc, tool_map, log_enabled)
            diagnostics["tool_call_details"].append(tool_detail)
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
            if is_error:
                diagnostics["tool_errors_total"] = int(diagnostics["tool_errors_total"]) + 1
                tool_errors_consecutive += 1
                diagnostics["tool_errors_consecutive_peak"] = max(
                    int(diagnostics["tool_errors_consecutive_peak"]),
                    tool_errors_consecutive,
                )
                if error_message:
                    diagnostics["tool_error_messages"].append(error_message)
            else:
                tool_errors_consecutive = 0

            if _threshold_reached():
                diagnostics["threshold_hit"] = True
                reason = (
                    f"工具失败达到阈值（counting={tool_error_counting}, "
                    f"max_tool_errors={max_tool_errors}）。"
                )
                diagnostics["threshold_reason"] = reason
                diagnostics["llm_usage"] = {
                    "prompt_tokens": usage_prompt,
                    "completion_tokens": usage_completion,
                    "total_tokens": usage_total,
                }
                diagnostics["llm_usage_complete"] = usage_complete
                return f"{error_marker_prefix} {reason}", diagnostics

    if log_enabled:
        logger.warning("达到最大工具调用轮次 ({})，生成最终响应", max_rounds)
    final = llm_with_tools.invoke(messages)
    model_name = _resolve_model_name_from_response(
        final,
        fallback=str(getattr(llm, "model_name", "unknown") or "unknown"),
    )
    diagnostics["model_name"] = model_name
    final_usage = extract_usage_from_response(final)
    if final_usage is None:
        usage_complete = False
    else:
        usage_prompt += final_usage["prompt_tokens"]
        usage_completion += final_usage["completion_tokens"]
        usage_total += final_usage["total_tokens"]
    diagnostics["llm_usage"] = {
        "prompt_tokens": usage_prompt,
        "completion_tokens": usage_completion,
        "total_tokens": usage_total,
    }
    diagnostics["llm_usage_complete"] = usage_complete
    return extract_text(final), diagnostics
