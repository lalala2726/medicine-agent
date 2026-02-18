from __future__ import annotations

from typing import Any, Mapping, Sequence

from loguru import logger

from app.schemas.admin_message import TokenUsage, TokenUsageBreakdownItem
from app.utils.token_utills import TokenUtils


def _to_non_negative_int(value: Any) -> int | None:
    """将输入值转换为非负整数，无法转换时返回 None。"""

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
    """按优先级读取整数值。"""

    for key in keys:
        if key not in payload:
            continue
        resolved = _to_non_negative_int(payload.get(key))
        if resolved is not None:
            return resolved
    return None


def _normalize_breakdown_item(raw_item: Mapping[str, Any]) -> TokenUsageBreakdownItem | None:
    """归一化单条 breakdown 记录。"""

    node_name = str(raw_item.get("node_name") or "").strip()
    if not node_name:
        return None

    prompt_tokens = _resolve_first_int(raw_item, ("prompt_tokens", "input_tokens")) or 0
    completion_tokens = _resolve_first_int(raw_item, ("completion_tokens", "output_tokens")) or 0
    raw_total_tokens = _resolve_first_int(raw_item, ("total_tokens",))
    total_tokens = prompt_tokens + completion_tokens
    if total_tokens == 0 and raw_total_tokens is not None:
        total_tokens = raw_total_tokens

    return TokenUsageBreakdownItem(
        node_name=node_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def _normalize_breakdown(
        breakdown: Sequence[TokenUsageBreakdownItem | Mapping[str, Any]] | None,
) -> list[TokenUsageBreakdownItem] | None:
    """归一化 breakdown 列表，自动忽略非法项。"""

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
        intermediate_tokens: int | None = None,
        total_tokens: int | None = None,
        breakdown: Sequence[TokenUsageBreakdownItem | Mapping[str, Any]] | None = None,
) -> TokenUsage:
    """
    统一构建 TokenUsage。

    规则：
    - 优先使用 prompt/completion 计算基础总量；
    - 若 prompt/completion 均缺失且传入 total_tokens，则回退到 total_tokens；
    - 总量口径统一为 prompt + completion + intermediate。
    """

    resolved_prompt = _to_non_negative_int(prompt_tokens) or 0
    resolved_completion = _to_non_negative_int(completion_tokens) or 0
    resolved_intermediate = (
        _to_non_negative_int(intermediate_tokens)
        if intermediate_tokens is not None
        else None
    )
    resolved_total = _to_non_negative_int(total_tokens)

    base_total = resolved_prompt + resolved_completion
    if base_total == 0 and resolved_total is not None:
        base_total = resolved_total
    if resolved_intermediate is not None:
        base_total += resolved_intermediate

    return TokenUsage(
        prompt_tokens=resolved_prompt,
        completion_tokens=resolved_completion,
        total_tokens=base_total,
        intermediate_tokens=resolved_intermediate,
        breakdown=_normalize_breakdown(breakdown),
    )


def normalize_token_usage(token_usage: TokenUsage | Mapping[str, Any] | None) -> TokenUsage | None:
    """归一化任意来源的 token usage 结构。"""

    if token_usage is None:
        return None
    if isinstance(token_usage, TokenUsage):
        return build_token_usage(
            prompt_tokens=token_usage.prompt_tokens,
            completion_tokens=token_usage.completion_tokens,
            intermediate_tokens=token_usage.intermediate_tokens,
            total_tokens=token_usage.total_tokens,
            breakdown=token_usage.breakdown,
        )
    if not isinstance(token_usage, Mapping):
        return None

    prompt_tokens = _resolve_first_int(token_usage, ("prompt_tokens", "input_tokens")) or 0
    completion_tokens = _resolve_first_int(token_usage, ("completion_tokens", "output_tokens")) or 0
    intermediate_tokens = _resolve_first_int(token_usage, ("intermediate_tokens",))
    total_tokens = _resolve_first_int(token_usage, ("total_tokens",))
    raw_breakdown = token_usage.get("breakdown")

    breakdown: Sequence[TokenUsageBreakdownItem | Mapping[str, Any]] | None = None
    if isinstance(raw_breakdown, list):
        breakdown = raw_breakdown

    return build_token_usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        intermediate_tokens=intermediate_tokens,
        total_tokens=total_tokens,
        breakdown=breakdown,
    )


def _estimate_tokens(
        *,
        text: str,
        model_name: str | None = None,
) -> int | None:
    """用 tiktoken 估算文本 token，失败时只记录日志。"""

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
    """估算 prompt/completion token（仅用于 usage 缺失兜底）。"""

    prompt_tokens = _estimate_tokens(text=prompt_text, model_name=model_name)
    completion_tokens = _estimate_tokens(text=completion_text, model_name=model_name)
    if prompt_tokens is None or completion_tokens is None:
        return None
    return build_token_usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        intermediate_tokens=None,
    )


def build_user_token_usage(
        *,
        content: str,
        model_name: str | None = None,
) -> TokenUsage:
    """构建 user 消息 token usage（仅记录输入 token）。"""

    estimated = estimate_prompt_completion_usage(
        prompt_text=content,
        completion_text="",
        model_name=model_name,
    )
    if estimated is None:
        return build_token_usage(
            prompt_tokens=0,
            completion_tokens=0,
            intermediate_tokens=None,
        )
    return build_token_usage(
        prompt_tokens=estimated.prompt_tokens,
        completion_tokens=0,
        intermediate_tokens=None,
    )


def merge_assistant_token_usage(
        *,
        stream_token_usage: TokenUsage | Mapping[str, Any] | None,
        prompt_text: str,
        completion_text: str,
        model_name: str | None = None,
) -> TokenUsage:
    """
    归并 assistant 的 token usage。

    优先使用流式过程采集到的真实 usage，缺失部分再回退 tiktoken 估算。
    """

    normalized_stream = normalize_token_usage(stream_token_usage)

    prompt_tokens = 0
    completion_tokens = 0
    intermediate_tokens = 0
    breakdown = None
    if normalized_stream is not None:
        prompt_tokens = normalized_stream.prompt_tokens
        completion_tokens = normalized_stream.completion_tokens
        intermediate_tokens = normalized_stream.intermediate_tokens or 0
        breakdown = normalized_stream.breakdown

    if prompt_tokens == 0 or completion_tokens == 0:
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
        intermediate_tokens=intermediate_tokens,
        breakdown=breakdown,
    )

