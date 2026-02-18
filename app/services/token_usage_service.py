from __future__ import annotations

from typing import Any, Mapping, Sequence

from loguru import logger

from app.schemas.admin_message import TokenUsage, TokenUsageBreakdownItem
from app.utils.token_utills import TokenUtils

UNKNOWN_MODEL_NAME = "unknown"


def _to_non_negative_int(value: Any) -> int | None:
    """
    将输入值转换为非负整数。

    Args:
        value: 任意输入值（如 int/str/float）。

    Returns:
        int | None: 转换成功返回非负整数；无法转换或小于 0 返回 None。
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


def _resolve_first_int(payload: Mapping[str, Any], keys: Sequence[str]) -> int | None:
    """
    按键优先级读取并解析整数值。

    Args:
        payload: 待读取的映射对象。
        keys: 需要尝试的键名列表（按优先级从前到后）。

    Returns:
        int | None: 命中可解析的非负整数时返回该值，否则返回 None。
    """

    for key in keys:
        if key not in payload:
            continue
        resolved = _to_non_negative_int(payload.get(key))
        if resolved is not None:
            return resolved
    return None


def _normalize_breakdown_item(raw_item: Mapping[str, Any]) -> TokenUsageBreakdownItem | None:
    """
    归一化单条 breakdown 记录。

    Args:
        raw_item: 原始节点统计数据。

    Returns:
        TokenUsageBreakdownItem | None: 合法节点返回归一化结果；缺少 node_name 时返回 None。
    """

    node_name = str(raw_item.get("node_name") or "").strip()
    if not node_name:
        return None

    model_name = str(raw_item.get("model_name") or UNKNOWN_MODEL_NAME).strip() or UNKNOWN_MODEL_NAME
    prompt_tokens = _resolve_first_int(raw_item, ("prompt_tokens", "input_tokens")) or 0
    completion_tokens = _resolve_first_int(raw_item, ("completion_tokens", "output_tokens")) or 0
    raw_total_tokens = _resolve_first_int(raw_item, ("total_tokens",))
    total_tokens = prompt_tokens + completion_tokens
    if total_tokens == 0 and raw_total_tokens is not None:
        total_tokens = raw_total_tokens

    return TokenUsageBreakdownItem(
        node_name=node_name,
        model_name=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def _normalize_breakdown(
        breakdown: Sequence[TokenUsageBreakdownItem | Mapping[str, Any]] | None,
) -> list[TokenUsageBreakdownItem] | None:
    """
    归一化 breakdown 列表，自动忽略非法项。

    Args:
        breakdown: 原始 breakdown 列表，可混合模型对象和字典。

    Returns:
        list[TokenUsageBreakdownItem] | None: 归一化后的列表；无有效数据时返回 None。
    """

    if breakdown is None:
        return None

    normalized_items: list[TokenUsageBreakdownItem] = []
    for item in breakdown:
        if isinstance(item, TokenUsageBreakdownItem):
            normalized_items.append(item)
            continue
        if not isinstance(item, Mapping):
            continue
        normalized_item = _normalize_breakdown_item(item)
        if normalized_item is not None:
            normalized_items.append(normalized_item)

    return normalized_items or None


def build_token_usage(
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int | None = None,
        breakdown: Sequence[TokenUsageBreakdownItem | Mapping[str, Any]] | None = None,
) -> TokenUsage:
    """
    统一构建 TokenUsage。

    总量口径固定为 prompt + completion；当两者都缺失时，允许回退到显式 total_tokens。

    Args:
        prompt_tokens: 输入 token 数。
        completion_tokens: 输出 token 数。
        total_tokens: 可选显式总量（仅在 prompt/completion 都为 0 时生效）。
        breakdown: 可选节点明细列表。

    Returns:
        TokenUsage: 归一化后的 token 使用结构。
    """

    resolved_prompt = _to_non_negative_int(prompt_tokens) or 0
    resolved_completion = _to_non_negative_int(completion_tokens) or 0
    resolved_total = _to_non_negative_int(total_tokens)

    normalized_total = resolved_prompt + resolved_completion
    if normalized_total == 0 and resolved_total is not None:
        normalized_total = resolved_total

    return TokenUsage(
        prompt_tokens=resolved_prompt,
        completion_tokens=resolved_completion,
        total_tokens=normalized_total,
        breakdown=_normalize_breakdown(breakdown),
    )


def normalize_token_usage(token_usage: TokenUsage | Mapping[str, Any] | None) -> TokenUsage | None:
    """
    归一化任意来源的 token usage 结构。

    Args:
        token_usage: 可能来自模型返回、业务传参或数据库读出的 usage 对象。

    Returns:
        TokenUsage | None: 归一化成功返回 TokenUsage；输入为空/非法返回 None。
    """

    if token_usage is None:
        return None
    if isinstance(token_usage, TokenUsage):
        return build_token_usage(
            prompt_tokens=token_usage.prompt_tokens,
            completion_tokens=token_usage.completion_tokens,
            total_tokens=token_usage.total_tokens,
            breakdown=token_usage.breakdown,
        )
    if not isinstance(token_usage, Mapping):
        return None

    prompt_tokens = _resolve_first_int(token_usage, ("prompt_tokens", "input_tokens")) or 0
    completion_tokens = _resolve_first_int(token_usage, ("completion_tokens", "output_tokens")) or 0
    total_tokens = _resolve_first_int(token_usage, ("total_tokens",))

    raw_breakdown = token_usage.get("breakdown")
    breakdown: Sequence[TokenUsageBreakdownItem | Mapping[str, Any]] | None = None
    if isinstance(raw_breakdown, list):
        breakdown = raw_breakdown

    return build_token_usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        breakdown=breakdown,
    )


def _estimate_tokens(
        *,
        text: str,
        model_name: str | None = None,
) -> int | None:
    """
    用 tiktoken 估算文本 token。

    Args:
        text: 待估算文本。
        model_name: 可选模型名，用于选择更准确编码器。

    Returns:
        int | None: 估算成功返回 token 数；失败仅记录日志并返回 None。
    """

    try:
        return TokenUtils.count_tokens(
            str(text or ""),
            model_name=model_name,
        )
    except Exception as exc:  # pragma: no cover - 防御性兜底
        logger.opt(exception=exc).warning("Failed to estimate tokens with tiktoken.")
        return None


def estimate_prompt_completion_usage(
        *,
        prompt_text: str,
        completion_text: str,
        model_name: str | None = None,
) -> TokenUsage | None:
    """
    估算 prompt/completion token（仅用于 usage 缺失兜底）。

    Args:
        prompt_text: 输入文本。
        completion_text: 输出文本。
        model_name: 可选模型名。

    Returns:
        TokenUsage | None: 估算成功返回 usage；任一侧估算失败返回 None。
    """

    prompt_tokens = _estimate_tokens(text=prompt_text, model_name=model_name)
    completion_tokens = _estimate_tokens(text=completion_text, model_name=model_name)
    if prompt_tokens is None or completion_tokens is None:
        return None
    return build_token_usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def build_user_token_usage(
        *,
        content: str,
        model_name: str | None = None,
) -> TokenUsage:
    """
    构建 user 消息 token usage（仅记录输入 token）。

    Args:
        content: 用户输入文本。
        model_name: 可选模型名。

    Returns:
        TokenUsage: user 消息 usage；估算失败时返回 0 值 usage。
    """

    estimated = estimate_prompt_completion_usage(
        prompt_text=content,
        completion_text="",
        model_name=model_name,
    )
    if estimated is None:
        return build_token_usage(
            prompt_tokens=0,
            completion_tokens=0,
        )
    return build_token_usage(
        prompt_tokens=estimated.prompt_tokens,
        completion_tokens=0,
    )


def merge_assistant_token_usage(
        *,
        stream_token_usage: TokenUsage | Mapping[str, Any] | None,
        prompt_text: str,
        completion_text: str,
        model_name: str | None = None,
) -> TokenUsage | None:
    """
    归并 assistant 的 token usage。

    优先使用流式过程采集到的真实 usage；缺失字段再回退 tiktoken 估算。
    当真实 usage 和估算均不可用时返回 None。

    Args:
        stream_token_usage: 流式采集到的 usage（可为空）。
        prompt_text: 本轮 prompt 文本（用于兜底估算）。
        completion_text: 本轮 completion 文本（用于兜底估算）。
        model_name: 可选模型名。

    Returns:
        TokenUsage | None: 合并后的 usage；真实与估算都失败时返回 None。
    """

    normalized_stream = normalize_token_usage(stream_token_usage)
    if normalized_stream is not None:
        prompt_tokens = normalized_stream.prompt_tokens
        completion_tokens = normalized_stream.completion_tokens

        if prompt_tokens > 0 and completion_tokens > 0:
            return normalized_stream

        estimated = estimate_prompt_completion_usage(
            prompt_text=prompt_text,
            completion_text=completion_text,
            model_name=model_name,
        )
        if estimated is not None:
            if prompt_tokens == 0:
                prompt_tokens = estimated.prompt_tokens
            if completion_tokens == 0:
                completion_tokens = estimated.completion_tokens
            return build_token_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                breakdown=normalized_stream.breakdown,
            )

        if normalized_stream.total_tokens > 0 or normalized_stream.breakdown:
            return normalized_stream
        return None

    estimated = estimate_prompt_completion_usage(
        prompt_text=prompt_text,
        completion_text=completion_text,
        model_name=model_name,
    )
    if estimated is None:
        return None
    return build_token_usage(
        prompt_tokens=estimated.prompt_tokens,
        completion_tokens=estimated.completion_tokens,
    )
