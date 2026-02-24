from __future__ import annotations

from typing import Any

from app.core.agent.agent_tool_trace import extract_text as _extract_text


def extract_text(message: Any) -> str:
    """
    从 LangChain 消息对象中提取纯文本内容。

    该函数是对 core 层 `extract_text` 的轻量转发，保留旧导入路径兼容。

    Args:
        message: LangChain 消息对象（AIMessage、AIMessageChunk 等）。

    Returns:
        str: 提取到的文本内容。
    """

    return _extract_text(message)
