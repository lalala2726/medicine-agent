from __future__ import annotations

import json
from typing import Any, Mapping

from langchain_core.messages import ToolMessage


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
