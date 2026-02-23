from __future__ import annotations

import asyncio
import contextvars
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


def agent_invoke(
    agent_instance: Any,
    history_messages: list[Any] | str,
) -> Any:
    """
    执行 agent 的 invoke 调用（优先异步 ainvoke）。

    Args:
        agent_instance: `create_agent_instance` 返回的 agent。
        history_messages: 输入消息列表。

    Returns:
        Any: agent 返回结果。
    """

    payload = {"messages": _normalize_history_messages(history_messages)}
    ainvoke = getattr(agent_instance, "ainvoke", None)
    if callable(ainvoke):
        return _run_async(ainvoke(payload))
    return agent_instance.invoke(payload)


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
