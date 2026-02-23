from __future__ import annotations

import contextvars
from typing import Any, Optional, Sequence

from app.core.agent_trace import run_model_with_trace

# 工具调用最大轮次，防止无限循环
MAX_TOOL_ROUNDS = 20

_TOOL_TRACE_STACK: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar(
    "admin_tool_trace_stack",
    default=None,
)


def extract_text(message: Any) -> str:
    """
    从 LangChain 消息对象中提取纯文本内容。

    Args:
        message: LangChain 消息对象（AIMessage、AIMessageChunk 等）

    Returns:
        str: 提取的文本内容，无内容时返回空字符串
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


def invoke(
        llm: Any,
        messages: list[Any],
        *,
        tools: Optional[Sequence[Any]] = None,
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
) -> str:
    """
    统一的 LLM 调用入口。

    流式输出由 LangGraph astream 在外层自动完成，节点内部统一走非流式调用。

    Args:
        llm: LangChain ChatModel 实例
        messages: 消息列表
        tools: 工具列表，传入后自动进入 Agent 模式（bind_tools + 工具调用循环）
        max_tool_rounds: 工具调用最大轮次，默认 5

    Returns:
        str: LLM 生成的完整文本
    """
    result = invoke_with_trace(
        llm,
        messages,
        tools=tools,
        max_tool_rounds=max_tool_rounds,
    )
    return str(result.get("text") or "")


def invoke_with_trace(
        llm: Any,
        messages: list[Any],
        *,
        tools: Optional[Sequence[Any]] = None,
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
) -> dict[str, Any]:
    """
    兼容入口：转发到 core 层的 `run_model_with_trace`。
    """
    return run_model_with_trace(
        llm,
        messages,
        tools=tools,
        max_tool_rounds=max_tool_rounds,
    )
