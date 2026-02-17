from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.admin.agent_state import ChatHistoryMessage


def _extract_message_text(message: ChatHistoryMessage) -> str:
    """
    提取历史消息中的纯文本内容。

    仅支持 `HumanMessage` 与 `AIMessage`，不兼容旧字典结构。
    """

    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return "" if content is None else str(content)


def _resolve_role(message: ChatHistoryMessage) -> str:
    """
    将 LangChain 历史消息对象映射为 role 字符串。
    """

    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    raise TypeError(f"Unsupported history message type: {type(message)!r}")


def history_to_prompt_lines(messages: list[ChatHistoryMessage]) -> str:
    """
    将历史消息转换为可读的多行 prompt 文本。
    """

    lines: list[str] = []
    for message in messages:
        text = _extract_message_text(message).strip()
        if not text:
            continue
        lines.append(f"{_resolve_role(message)}: {text}")
    return "\n".join(lines)


def history_to_role_dicts(messages: list[ChatHistoryMessage]) -> list[dict[str, str]]:
    """
    将历史消息转换为统一的 `role/content` 字典列表，供 JSON payload 使用。
    """

    history_items: list[dict[str, str]] = []
    for message in messages:
        text = _extract_message_text(message).strip()
        if not text:
            continue
        history_items.append(
            {
                "role": _resolve_role(message),
                "content": text,
            }
        )
    return history_items


def build_messages_with_history(
        *,
        system_prompt: str,
        history_messages: list[ChatHistoryMessage],
        fallback_user_input: str,
) -> list[Any]:
    """
    构建模型输入消息。

    规则：
    1. 有历史消息时：`SystemMessage + 历史消息`；
    2. 无历史消息时：`SystemMessage + HumanMessage(fallback_user_input)`。
    """

    messages: list[Any] = [SystemMessage(content=system_prompt)]
    if history_messages:
        messages.extend(history_messages)
        return messages

    fallback_input = fallback_user_input.strip()
    if fallback_input:
        messages.append(HumanMessage(content=fallback_input))
    return messages


__all__ = [
    "build_messages_with_history",
    "history_to_prompt_lines",
    "history_to_role_dicts",
]
