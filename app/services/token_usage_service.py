from __future__ import annotations

from typing import Any, Mapping, Sequence

from app.agent.admin.state import (
    ExecutionTraceState,
    NodeTokenBreakdownState,
    TokenUsageState,
    ToolLlmBreakdownState,
)


def _to_non_negative_int(value: Any) -> int | None:
    """
    将任意值转换为非负整数。

    Args:
        value: 待转换的原始值。

    Returns:
        int | None: 转换成功返回非负整数；无法转换或为负数时返回 None。
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


def _normalize_usage_payload(raw: Mapping[str, Any] | None) -> dict[str, int] | None:
    """
    标准化 provider usage 结构。

    支持两套字段：
    - `prompt_tokens/completion_tokens/total_tokens`
    - `input_tokens/output_tokens/total_tokens`

    Args:
        raw: 原始 usage 数据。

    Returns:
        dict[str, int] | None: 标准化后的 usage；无有效 token 字段时返回 None。
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


def _zero_usage() -> dict[str, int]:
    """
    返回全 0 usage 结构。

    Returns:
        dict[str, int]: `prompt/completion/total` 均为 0 的结构。
    """

    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def sum_usage(usages: Sequence[Mapping[str, Any] | None]) -> dict[str, int]:
    """
    累加 usage 列表。

    Args:
        usages: usage 列表，允许包含 None 或不合法项。

    Returns:
        dict[str, int]: 累加后的 `prompt/completion/total`。
    """

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    for usage in usages:
        normalized = _normalize_usage_payload(usage)
        if normalized is None:
            continue
        prompt_tokens += normalized["prompt_tokens"]
        completion_tokens += normalized["completion_tokens"]
        total_tokens += normalized["total_tokens"]
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def merge_node_and_tool_usage(
    node_usage: Mapping[str, Any] | None,
    tool_usage: Mapping[str, Any] | None,
) -> dict[str, int]:
    """
    合并节点 usage 与工具 usage。

    Args:
        node_usage: 节点自身 usage。
        tool_usage: 节点下工具 usage 汇总。

    Returns:
        dict[str, int]: 合并后的 usage 结果。
    """

    return sum_usage([node_usage, tool_usage])


def _build_tool_breakdown(
    tool_calls: Sequence[Mapping[str, Any]] | None,
) -> tuple[list[ToolLlmBreakdownState], dict[str, int], bool]:
    """
    递归构建工具级 token 明细。

    Args:
        tool_calls: 节点工具调用列表（支持 children 递归）。

    Returns:
        tuple[list[ToolLlmBreakdownState], dict[str, int], bool]:
            - 工具级 token 明细；
            - 工具 token 汇总；
            - usage 是否完整（任意 LLM 调用缺 usage 则为 False）。
    """

    if not tool_calls:
        return [], _zero_usage(), True

    breakdown: list[ToolLlmBreakdownState] = []
    all_prompt = 0
    all_completion = 0
    all_total = 0
    is_complete = True

    for item in tool_calls:
        if not isinstance(item, Mapping):
            continue

        tool_name = str(item.get("tool_name") or "").strip() or "unknown_tool"
        tool_input = item.get("tool_input")
        llm_used = bool(item.get("llm_used"))
        llm_usage_complete = bool(item.get("llm_usage_complete", True))
        llm_token_usage = item.get("llm_token_usage")
        own_usage = (
            _normalize_usage_payload(llm_token_usage)
            if isinstance(llm_token_usage, Mapping)
            else None
        )
        if llm_used and own_usage is None:
            llm_usage_complete = False
        own_usage = own_usage or _zero_usage()

        raw_children = item.get("children")
        children_breakdown, children_usage, children_complete = _build_tool_breakdown(
            raw_children if isinstance(raw_children, list) else None
        )
        tool_usage = merge_node_and_tool_usage(own_usage, children_usage)
        breakdown.append(
            ToolLlmBreakdownState(
                tool_name=tool_name,
                tool_input=tool_input,
                prompt_tokens=tool_usage["prompt_tokens"],
                completion_tokens=tool_usage["completion_tokens"],
                total_tokens=tool_usage["total_tokens"],
                children=children_breakdown,
            )
        )

        all_prompt += tool_usage["prompt_tokens"]
        all_completion += tool_usage["completion_tokens"]
        all_total += tool_usage["total_tokens"]
        is_complete = is_complete and children_complete and (not llm_used or llm_usage_complete)

    return (
        breakdown,
        {
            "prompt_tokens": all_prompt,
            "completion_tokens": all_completion,
            "total_tokens": all_total,
        },
        is_complete,
    )


def build_token_usage_from_execution_traces(
    execution_traces: Sequence[Mapping[str, Any]] | None,
) -> TokenUsageState | None:
    """
    由 execution_traces 构建消息级 token 使用汇总。

    Args:
        execution_traces: workflow state 中累计的节点执行轨迹。

    Returns:
        TokenUsageState | None:
            - 有效节点存在时返回消息级 token 汇总；
            - 轨迹为空或无有效节点时返回 None。
    """

    if not execution_traces:
        return None

    node_breakdown: list[NodeTokenBreakdownState] = []
    message_prompt = 0
    message_completion = 0
    message_total = 0
    is_complete = True

    for trace in execution_traces:
        if not isinstance(trace, Mapping):
            continue

        node_name = str(trace.get("node_name") or "").strip()
        if not node_name:
            continue
        model_name = str(trace.get("model_name") or "").strip() or "unknown"

        llm_used = bool(trace.get("llm_used", True))
        llm_usage_complete = bool(trace.get("llm_usage_complete", True))
        raw_node_usage = trace.get("llm_token_usage")
        node_usage = (
            _normalize_usage_payload(raw_node_usage)
            if isinstance(raw_node_usage, Mapping)
            else None
        )
        if llm_used and node_usage is None:
            llm_usage_complete = False
        node_usage = node_usage or _zero_usage()

        raw_tool_calls = trace.get("tool_calls")
        tools_breakdown, tools_usage, tools_complete = _build_tool_breakdown(
            raw_tool_calls if isinstance(raw_tool_calls, list) else None
        )
        node_breakdown.append(
            NodeTokenBreakdownState(
                node_name=node_name,
                model_name=model_name,
                prompt_tokens=node_usage["prompt_tokens"],
                completion_tokens=node_usage["completion_tokens"],
                total_tokens=node_usage["total_tokens"],
                tool_tokens_total=tools_usage["total_tokens"],
                tool_llm_breakdown=tools_breakdown,
            )
        )

        message_prompt += node_usage["prompt_tokens"] + tools_usage["prompt_tokens"]
        message_completion += node_usage["completion_tokens"] + tools_usage["completion_tokens"]
        message_total += node_usage["total_tokens"] + tools_usage["total_tokens"]
        is_complete = is_complete and tools_complete and (not llm_used or llm_usage_complete)

    if not node_breakdown:
        return None

    return TokenUsageState(
        prompt_tokens=message_prompt,
        completion_tokens=message_completion,
        total_tokens=message_total,
        is_complete=is_complete,
        node_breakdown=node_breakdown,
    )


def append_trace_and_refresh_token_usage(
    execution_traces: Sequence[ExecutionTraceState] | None,
    trace_item: ExecutionTraceState,
) -> tuple[list[ExecutionTraceState], TokenUsageState | None]:
    """
    追加单条节点 trace，并同步刷新 state.token_usage。

    Args:
        execution_traces: 当前 state 中已有的执行轨迹列表。
        trace_item: 当前节点新产生的执行轨迹。

    Returns:
        tuple[list[ExecutionTraceState], TokenUsageState | None]:
            - 新的 execution_traces；
            - 基于新轨迹列表实时计算的 token_usage。
    """

    traces = list(execution_traces or [])
    traces.append(trace_item)
    token_usage = build_token_usage_from_execution_traces(traces)
    return traces, token_usage


def _normalize_state_token_usage(raw: Mapping[str, Any] | None) -> TokenUsageState | None:
    """
    校验并标准化 state 中缓存的 token_usage。

    Args:
        raw: state.token_usage 原始值。

    Returns:
        TokenUsageState | None: 标准化后的 token_usage；非法时返回 None。
    """

    if raw is None:
        return None

    normalized_usage = _normalize_usage_payload(raw)
    if normalized_usage is None:
        return None

    raw_breakdown = raw.get("node_breakdown")
    node_breakdown = raw_breakdown if isinstance(raw_breakdown, list) else []

    return TokenUsageState(
        prompt_tokens=normalized_usage["prompt_tokens"],
        completion_tokens=normalized_usage["completion_tokens"],
        total_tokens=normalized_usage["total_tokens"],
        is_complete=bool(raw.get("is_complete", True)),
        node_breakdown=node_breakdown,
    )


def resolve_persistable_token_usage(
    state_token_usage: Mapping[str, Any] | None,
    execution_traces: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any] | None:
    """
    生成可持久化的 token_usage。

    规则：
    1. 优先使用 state 中已经聚合好的 `token_usage`；
    2. 若 state 中缺失或非法，则基于 `execution_traces` 兜底重建；
    3. 不做 token 估算，仅使用已返回 usage 聚合结果。

    Args:
        state_token_usage: state 中缓存的消息级 token_usage。
        execution_traces: state 中累计的 execution_traces（用于兜底重建）。

    Returns:
        dict[str, Any] | None: 可直接传给 `add_message(..., token_usage=...)` 的结构。
    """

    normalized_state_usage = (
        _normalize_state_token_usage(state_token_usage)
        if isinstance(state_token_usage, Mapping)
        else None
    )
    if normalized_state_usage is not None:
        return dict(normalized_state_usage)

    rebuilt = build_token_usage_from_execution_traces(execution_traces)
    if rebuilt is None:
        return None
    return dict(rebuilt)


def build_message_token_usage(
    execution_traces: Sequence[Mapping[str, Any]] | None,
) -> TokenUsageState | None:
    """
    兼容旧函数名：由 execution_trace 构建消息级 token usage。

    Args:
        execution_traces: 节点执行轨迹。

    Returns:
        TokenUsageState | None: 与 `build_token_usage_from_execution_traces` 等价。
    """

    return build_token_usage_from_execution_traces(execution_traces)
