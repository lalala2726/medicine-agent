"""
LLM 执行与工具调用辅助模块（execution utils）。

提供统一的 LLM 调用接口，支持：
- 普通文本生成
- 带工具调用的 Agent 模式

职责边界：
- 这里负责“如何执行模型/工具调用”（invoke、tool loop、工具执行）；
- 不负责 SSE 输出编排、事件队列消费、HTTP streaming 封包。
  这类逻辑应放在 `app/services/assistant_stream_service.py`。

放置建议：
- 与模型调用/工具调用强相关的执行逻辑放在这里；
- 与流式响应协议（SSE）强相关的传输逻辑不要放这里。

流式输出说明：
    节点内部统一使用同步调用（llm.invoke），逐 token 的流式推送由
    LangGraph 的 astream(stream_mode=["messages"]) 在外层自动完成。
    节点无需关心流式细节。

    关键：LLM 调用必须在 LangGraph 管理的线程上下文中执行（即使用
    同步 llm.invoke），否则 LangGraph 无法通过 context variables
    拦截 token 流。工具执行因为是 async 函数，会通过辅助线程运行，
    但这不影响 LLM token 的流式推送。
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
from typing import Any, Optional, Sequence

from langchain_core.messages import ToolMessage
from loguru import logger

from app.agent.admin.agent_state import AgentState
from app.core.assistant_status import emit_function_call, resolve_tool_call_messages

# 工具调用最大轮次，防止无限循环
MAX_TOOL_ROUNDS = 5


def _is_tool_log_enabled() -> bool:
    """
    检查是否启用工具调用日志。

    Returns:
        bool: AGENT_TOOL_LOG_ENABLED 为 true 时返回 True
    """
    value = os.getenv("AGENT_TOOL_LOG_ENABLED", "false").strip().lower()
    return value in {"1", "true", "yes", "on"}


def has_plan(state: AgentState | dict[str, Any]) -> bool:
    """
    检查状态中是否存在执行计划。

    Args:
        state: Agent 状态字典

    Returns:
        bool: 存在非空 plan 列表时返回 True
    """
    raw_plan = state.get("plan")
    return isinstance(raw_plan, list) and len(raw_plan) > 0


def is_final_node(state: AgentState | dict[str, Any], node_name: str) -> bool:
    """
    判断当前节点是否为最终输出节点。

    判断条件（按优先级）：
    1. 当前 step 显式标记 final_output=true
    2. gateway_router 直达该节点且无 plan
    3. 兼容旧逻辑：planner 标记为最后阶段，且 next_nodes 仅包含该节点

    Args:
        state: Agent 状态字典
        node_name: 当前节点名称

    Returns:
        bool: 是最终节点返回 True
    """
    routing = state.get("routing") or {}

    current_step_map = routing.get("current_step_map") or {}
    if isinstance(current_step_map, dict):
        step = current_step_map.get(node_name)
        if isinstance(step, dict) and bool(step.get("final_output")):
            return True

    route_target = routing.get("route_target")
    if route_target == node_name and not has_plan(state):
        return True

    next_nodes = routing.get("next_nodes")
    return (
            bool(routing.get("is_final_stage"))
            and isinstance(next_nodes, list)
            and len(next_nodes) == 1
            and next_nodes[0] == node_name
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
    if tools:
        return _invoke_with_tools(llm, messages, tools, max_tool_rounds)

    response = llm.invoke(messages)
    return extract_text(response)


def invoke_with_optional_stream(
        llm: Any,
        messages: list[Any],
        *,
        tools: Optional[Sequence[Any]] = None,
        enable_stream: bool = False,
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
) -> tuple[str, list[str]]:
    """
    在允许时优先使用 stream 收集文本，否则回退到 invoke。

    Args:
        llm: LangChain ChatModel 实例
        messages: 消息列表
        tools: 工具列表；stream 与 invoke 都会复用该工具集合
        enable_stream: 是否启用 stream 分支
        max_tool_rounds: invoke 工具调用最大轮次

    Returns:
        tuple[str, list[str]]: (最终文本, 流式分片列表)
    """
    stream_chunks: list[str] = []
    if enable_stream:
        llm_for_stream = llm.bind_tools(tools) if tools else llm
        stream_fn = getattr(llm_for_stream, "stream", None)
        if callable(stream_fn):
            for chunk in stream_fn(messages):
                text = extract_text(chunk)
                if text:
                    stream_chunks.append(text)

    if stream_chunks:
        return "".join(stream_chunks), stream_chunks

    content = invoke(
        llm,
        messages,
        tools=tools,
        max_tool_rounds=max_tool_rounds,
    )
    return content, stream_chunks


def _invoke_with_tools(
        llm: Any,
        messages: list[Any],
        tools: Sequence[Any],
        max_rounds: int,
) -> str:
    """
    带工具调用的同步 Agent 模式。

    流程：bind_tools → llm.invoke → 检查 tool_calls → 执行工具 → 结果反馈 → 循环。

    关键设计：LLM 调用使用同步 llm.invoke()，保持在 LangGraph 管理的
    线程上下文中，确保 astream(stream_mode="messages") 能够拦截 token 流。
    工具执行（async 函数）通过 _run_async 在辅助线程中运行，不影响流式推送。

    Args:
        llm: LangChain ChatModel 实例
        messages: 消息列表（会被原地修改）
        tools: 工具列表
        max_rounds: 最大工具调用轮次

    Returns:
        str: LLM 最终生成的文本
    """
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}
    log_enabled = _is_tool_log_enabled()

    for round_idx in range(max_rounds):
        # 同步调用 — 保持 LangGraph 上下文，支持 token 流式拦截
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        tool_calls = getattr(response, "tool_calls", None)
        if not tool_calls:
            if log_enabled:
                logger.info("第 {} 轮无工具调用，返回最终结果", round_idx + 1)
            return extract_text(response)

        if log_enabled:
            logger.info(
                "第 {} 轮触发 {} 个工具调用", round_idx + 1, len(tool_calls)
            )

        for tc in tool_calls:
            result = _exec_tool(tc, tool_map, log_enabled)
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    if log_enabled:
        logger.warning("达到最大工具调用轮次 ({})，生成最终响应", max_rounds)
    final = llm_with_tools.invoke(messages)
    return extract_text(final)


def _run_async(coro: Any) -> Any:
    """
    在当前线程中运行一个 async 协程。

    如果当前线程已有 event loop 在运行（例如被 LangGraph 的 executor 管理），
    则在新线程中启动一个临时 event loop；否则直接 asyncio.run。

    Args:
        coro: 异步协程对象

    Returns:
        协程的返回值
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        # 透传当前 contextvars（如 Authorization、状态事件发射器）到线程池任务，
        # 否则工具执行线程拿不到请求上下文。
        current_context = contextvars.copy_context()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(current_context.run, asyncio.run, coro).result()
    return asyncio.run(coro)


def _exec_tool(
        tool_call: dict, tool_map: dict[str, Any], log_enabled: bool = None
) -> str:
    """
    执行单个工具调用。

    工具函数可能是 async 的（例如通过 httpx 做 HTTP 请求），
    通过 _run_async 统一处理。

    Args:
        tool_call: 工具调用信息，包含 name、args、id
        tool_map: 工具名称到工具函数的映射
        log_enabled: 是否启用日志（None 时自动检查环境变量）

    Returns:
        str: JSON 格式的执行结果或错误信息
    """
    name = tool_call["name"]
    args = tool_call.get("args", {})
    tool_fn = tool_map.get(name)
    tool_node = f"tool:{name}"

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
        return json.dumps({"error": unknown_message}, ensure_ascii=False)

    if log_enabled:
        logger.info("工具调用: name={} args={}", name, args)

    try:
        result = _run_async(tool_fn.ainvoke(args))
        if log_enabled:
            logger.info(
                "工具返回: name={} result={}",
                name,
                json.dumps(result, ensure_ascii=False, default=str)[:500],
            )
        return (
            json.dumps(result, ensure_ascii=False, default=str)
            if isinstance(result, (dict, list))
            else str(result)
        )
    except Exception as exc:
        if log_enabled:
            logger.error("工具执行失败: name={} error={}", name, exc)
        return json.dumps({"error": f"工具执行失败: {name}, {exc}"}, ensure_ascii=False)
