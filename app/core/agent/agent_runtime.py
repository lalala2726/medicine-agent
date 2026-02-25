from __future__ import annotations

import asyncio
import contextvars
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from langchain_core.messages import HumanMessage

from app.core.agent.agent_tool_trace import extract_text


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


def _normalize_history_messages(history_messages: list[Any] | str | None) -> list[Any]:
    """
    规范化运行层输入消息。

    Args:
        history_messages: 可传入消息列表或字符串。字符串会自动包装为 `HumanMessage`。

    Returns:
        list[Any]: 标准化后的消息列表。
    """

    if isinstance(history_messages, list):
        return list(history_messages)
    if history_messages is None:
        return []
    return [HumanMessage(content=str(history_messages))]


def _resolve_messages_from_payload(payload: Any) -> list[Any]:
    """
    从 agent 调用返回值中提取消息列表。

    Args:
        payload: `agent.invoke/ainvoke` 原始返回值。

    Returns:
        list[Any]: 消息列表；无法解析时返回空列表。
    """

    if not isinstance(payload, Mapping):
        return []
    raw_messages = payload.get("messages")
    if isinstance(raw_messages, list):
        return list(raw_messages)
    return []


def _resolve_content_from_messages(messages: list[Any]) -> tuple[str, Any]:
    """
    从消息序列中提取最后一条 AI 消息文本及原始 content。

    Args:
        messages: 消息列表。

    Returns:
        tuple[str, Any]:
            - 文本内容（已 strip）；
            - 原始 content。
    """

    for message in reversed(messages):
        if str(getattr(message, "type", "") or "").lower() != "ai":
            continue
        raw_content = getattr(message, "content", None)
        return extract_text(message).strip(), raw_content
    return "", None


@dataclass(frozen=True)
class AgentInvokeResult:
    """
    `agent_invoke` 的标准化返回结构。

    Attributes:
        payload: agent 原始返回值。
        messages: 从 payload 提取的消息列表。
        content: 优先最后一条 AI 文本，否则回退 payload 的 `output/text`。
        raw_content: 最后一条 AI 原始 content（若存在）。
    """

    payload: Any
    messages: list[Any]
    content: str
    raw_content: Any


def agent_invoke(
        agent_instance: Any,
        history_messages: list[Any] | str,
) -> AgentInvokeResult:
    """
    执行 agent 的 invoke 调用（优先异步 ainvoke）。

    Args:
        agent_instance: `create_agent_instance` 返回的 agent。
        history_messages: 输入消息列表。

    Returns:
        AgentInvokeResult: 标准化后的 invoke 结果。
    """

    payload = {"messages": _normalize_history_messages(history_messages)}
    ainvoke = getattr(agent_instance, "ainvoke", None)
    if callable(ainvoke):
        raw_result = _run_async(ainvoke(payload))
    else:
        raw_result = agent_instance.invoke(payload)

    messages = _resolve_messages_from_payload(raw_result)
    content, raw_content = _resolve_content_from_messages(messages)
    if not content and isinstance(raw_result, Mapping):
        content = str(raw_result.get("output") or raw_result.get("text") or "").strip()
        if raw_content is None:
            raw_content = raw_result.get("output") or raw_result.get("text")

    return AgentInvokeResult(
        payload=raw_result,
        messages=messages,
        content=content,
        raw_content=raw_content,
    )


def agent_stream(
        agent_instance: Any,
        history_messages: list[Any] | str,
        on_model_delta: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """
    执行 agent 的 astream 调用。

    Args:
        agent_instance: `create_agent_instance` 返回的 agent。
        history_messages: 输入消息列表。
        on_model_delta: 可选回调；当模型节点产出文本分片时触发。

    Returns:
        dict[str, Any]:
            - latest_state: 最后一次 values 状态。
            - streamed_text: 全部分片拼接文本。
            - final_messages: 从 latest_state 解析出的最终消息列表。
    """
    normalized_history_messages = _normalize_history_messages(history_messages)

    async def _collect_stream_events() -> tuple[list[str], dict[str, Any]]:
        streamed_chunks: list[str] = []
        latest_state: dict[str, Any] = {}

        async for raw_event in agent_instance.astream(
                {"messages": normalized_history_messages},
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
            if on_model_delta is not None:
                on_model_delta(delta)

        return streamed_chunks, latest_state

    streamed_chunks, latest_state = _run_async(_collect_stream_events())
    final_messages: list[Any] = []
    raw_messages = latest_state.get("messages")
    if isinstance(raw_messages, list):
        final_messages = list(raw_messages)

    return {
        "latest_state": latest_state,
        "streamed_text": "".join(streamed_chunks),
        "final_messages": final_messages,
    }
