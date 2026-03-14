from __future__ import annotations

from typing import Any

from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException


def normalize_question(question: str) -> str:
    """规范化并校验知识库检索问题。

    Args:
        question: 用户传入的原始问题文本。

    Returns:
        去除首尾空白后的问题文本。

    Raises:
        ServiceException: 当问题在去空白后为空时抛出。
    """

    normalized_question = str(question or "").strip()
    if not normalized_question:
        raise ServiceException(
            code=ResponseCode.BAD_REQUEST,
            message="question 不能为空",
        )
    return normalized_question


def coerce_optional_int(value: Any) -> int | None:
    """将元信息中的值规整为可选整数。

    Args:
        value: 待转换的原始值。

    Returns:
        转换成功时返回整数；无法转换时返回 ``None``。
    """

    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_message_content_text(content: Any) -> str:
    """从 LangChain 消息内容中提取纯文本。

    Args:
        content: LangChain 消息的 ``content`` 字段。

    Returns:
        规整后的纯文本内容；无法提取时返回空字符串。
    """

    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    pieces: list[str] = []
    for item in content:
        if isinstance(item, str) and item.strip():
            pieces.append(item.strip())
            continue
        if isinstance(item, dict):
            candidate = item.get("text")
            if isinstance(candidate, str) and candidate.strip():
                pieces.append(candidate.strip())
    return "\n".join(pieces).strip()


def strip_markdown_json_fence(text: str) -> str:
    """移除可能包裹 JSON 的 markdown 代码块围栏。

    Args:
        text: 可能包含 markdown 代码块围栏的文本。

    Returns:
        去掉围栏后的文本内容。
    """

    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def trim_content_to_budget(*, content: str, remaining_chars: int) -> str:
    """将片段内容裁剪到剩余预算以内。

    Args:
        content: 原始片段文本。
        remaining_chars: 当前剩余的字符预算。

    Returns:
        裁剪后的片段文本。
    """

    if remaining_chars <= 0:
        return ""
    if len(content) <= remaining_chars:
        return content
    if remaining_chars <= 3:
        return content[:remaining_chars]
    return f"{content[:remaining_chars - 3]}..."
