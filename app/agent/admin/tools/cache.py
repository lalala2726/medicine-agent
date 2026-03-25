"""
管理端工具缓存模块。

说明：
1. 管理端工具缓存按会话 UUID 隔离，避免不同会话串数据；
2. 缓存粒度为“工具名 + 规范化入参哈希”，避免同一工具不同筛选条件互相覆盖；
3. 缓存仅用于给模型注入可复用上下文，不在工具层直接短路调用。
"""

from __future__ import annotations

import hashlib
import json
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from app.core.database.redis import RedisHashCache

# 管理端工具缓存 Redis key 前缀。
ADMIN_TOOL_CACHE_KEY_PREFIX = "admin:tool_cache"
# 管理端工具缓存 TTL，单位秒。
ADMIN_TOOL_CACHE_TTL_SECONDS = 1800
# 注入 system prompt 的最大缓存记录数。
ADMIN_TOOL_CACHE_MAX_PROMPT_RECORDS = 8
# 管理端工具缓存提示词固定标题。
ADMIN_TOOL_CACHE_PROMPT_TITLE = "已缓存后台工具结果"

_ADMIN_TOOL_CACHE_CONVERSATION_UUID: ContextVar[str | None] = ContextVar(
    "admin_tool_cache_conversation_uuid",
    default=None,
)


class AdminToolCacheEntry(BaseModel):
    """
    功能描述：
        管理端单条工具缓存记录模型。

    参数说明：
        tool_name (str): 工具名称。
        tool_input (Any): 工具入参。
        tool_output (Any): 工具输出。
        updated_at (str): 缓存写入时间（UTC ISO8601）。

    返回值：
        无（数据模型定义）。

    异常说明：
        无。
    """

    tool_name: str
    tool_input: Any
    tool_output: Any
    updated_at: str


def _normalize_required_text(value: str, *, field_name: str) -> str:
    """
    功能描述：
        规范化必填字符串字段。

    参数说明：
        value (str): 原始字段值。
        field_name (str): 字段名称。

    返回值：
        str: 去除首尾空白后的非空字符串。

    异常说明：
        ValueError: 当字段为空时抛出。
    """

    normalized_value = str(value or "").strip()
    if not normalized_value:
        raise ValueError(f"{field_name} 不能为空")
    return normalized_value


def _serialize_cache_value(value: Any) -> Any:
    """
    功能描述：
        将缓存值递归转换为可 JSON 序列化结构。

    参数说明：
        value (Any): 原始缓存值。

    返回值：
        Any: 可 JSON 序列化的结构。

    异常说明：
        无。
    """

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, list):
        return [_serialize_cache_value(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize_cache_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _serialize_cache_value(item)
            for key, item in value.items()
        }
    return value


def _decode_redis_text(value: Any, *, field_name: str) -> str:
    """
    功能描述：
        将 Redis 返回值解码为文本。

    参数说明：
        value (Any): Redis 原始字段或字段值。
        field_name (str): 当前字段名称。

    返回值：
        str: 解码后的文本。

    异常说明：
        ValueError: 当解码结果为空时抛出。
    """

    if isinstance(value, bytes):
        return value.decode("utf-8")
    return _normalize_required_text(str(value or ""), field_name=field_name)


def _build_stable_json(value: Any) -> str:
    """
    功能描述：
        构造稳定排序的 JSON 文本，用于计算缓存哈希。

    参数说明：
        value (Any): 任意待序列化对象。

    返回值：
        str: 稳定排序后的 JSON 文本。

    异常说明：
        无。
    """

    serialized_value = _serialize_cache_value(value)
    return json.dumps(
        serialized_value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _build_admin_tool_cache_field(tool_name: str, tool_input: Any) -> str:
    """
    功能描述：
        构造 Redis Hash 字段名。

    参数说明：
        tool_name (str): 工具名称。
        tool_input (Any): 工具入参。

    返回值：
        str: 形如 `{tool_name}:{input_hash}` 的字段名。

    异常说明：
        ValueError: 当工具名称为空时抛出。
    """

    normalized_tool_name = _normalize_required_text(tool_name, field_name="tool_name")
    stable_json = _build_stable_json(tool_input)
    input_hash = hashlib.sha256(stable_json.encode("utf-8")).hexdigest()
    return f"{normalized_tool_name}:{input_hash}"


def build_admin_tool_cache_key(conversation_uuid: str) -> str:
    """
    功能描述：
        构造管理端工具缓存 Redis key。

    参数说明：
        conversation_uuid (str): 当前会话 UUID。

    返回值：
        str: Redis key。

    异常说明：
        ValueError: 当会话 UUID 为空时抛出。
    """

    normalized_uuid = _normalize_required_text(
        conversation_uuid,
        field_name="conversation_uuid",
    )
    return f"{ADMIN_TOOL_CACHE_KEY_PREFIX}:{normalized_uuid}"


def bind_current_admin_tool_cache_conversation(conversation_uuid: str) -> Token[str | None]:
    """
    功能描述：
        绑定当前上下文正在使用的会话 UUID。

    参数说明：
        conversation_uuid (str): 当前会话 UUID。

    返回值：
        Token[str | None]: `ContextVar` 重置令牌。

    异常说明：
        ValueError: 当会话 UUID 为空时抛出。
    """

    normalized_uuid = _normalize_required_text(
        conversation_uuid,
        field_name="conversation_uuid",
    )
    return _ADMIN_TOOL_CACHE_CONVERSATION_UUID.set(normalized_uuid)


def reset_current_admin_tool_cache_conversation(token: Token[str | None]) -> None:
    """
    功能描述：
        重置当前上下文绑定的会话 UUID。

    参数说明：
        token (Token[str | None]): 绑定时返回的重置令牌。

    返回值：
        None。

    异常说明：
        无。
    """

    _ADMIN_TOOL_CACHE_CONVERSATION_UUID.reset(token)


def _get_current_admin_tool_cache_conversation_uuid() -> str:
    """
    功能描述：
        读取当前上下文绑定的会话 UUID。

    参数说明：
        无。

    返回值：
        str: 当前会话 UUID。

    异常说明：
        ValueError: 当当前上下文未绑定会话 UUID 时抛出。
    """

    conversation_uuid = _ADMIN_TOOL_CACHE_CONVERSATION_UUID.get()
    return _normalize_required_text(
        str(conversation_uuid or ""),
        field_name="conversation_uuid",
    )


def load_admin_tool_cache(conversation_uuid: str) -> dict[str, Any]:
    """
    功能描述：
        读取当前会话的全部管理端工具缓存。

    参数说明：
        conversation_uuid (str): 当前会话 UUID。

    返回值：
        dict[str, Any]: `field -> cache_entry` 的缓存映射。

    异常说明：
        ValueError: 当会话 UUID 为空时抛出。
    """

    cache_key = build_admin_tool_cache_key(conversation_uuid)
    raw_hash = RedisHashCache().h_get_all(cache_key)
    if not raw_hash:
        return {}

    cache_payload: dict[str, Any] = {}
    for raw_field, raw_value in raw_hash.items():
        decoded_field = _decode_redis_text(raw_field, field_name="cache_field")
        cache_payload[decoded_field] = json.loads(
            _decode_redis_text(raw_value, field_name=decoded_field),
        )
    return cache_payload


def save_admin_tool_cache_entry(
        conversation_uuid: str,
        tool_name: str,
        tool_input: Any,
        tool_output: Any,
) -> None:
    """
    功能描述：
        保存单条管理端工具缓存记录。

    参数说明：
        conversation_uuid (str): 当前会话 UUID。
        tool_name (str): 工具名称。
        tool_input (Any): 工具入参。
        tool_output (Any): 工具输出。

    返回值：
        None。

    异常说明：
        ValueError: 当会话 UUID 或工具名称为空时抛出。
    """

    cache_key = build_admin_tool_cache_key(conversation_uuid)
    normalized_tool_name = _normalize_required_text(tool_name, field_name="tool_name")
    normalized_tool_input = _serialize_cache_value(tool_input)
    normalized_tool_output = _serialize_cache_value(tool_output)
    cache_field = _build_admin_tool_cache_field(
        normalized_tool_name,
        normalized_tool_input,
    )
    cache_entry = AdminToolCacheEntry(
        tool_name=normalized_tool_name,
        tool_input=normalized_tool_input,
        tool_output=normalized_tool_output,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    redis_hash_cache = RedisHashCache()
    redis_hash_cache.h_put(
        cache_key,
        cache_field,
        json.dumps(
            cache_entry.model_dump(mode="json"),
            ensure_ascii=False,
        ),
    )
    redis_hash_cache.expire(cache_key, ADMIN_TOOL_CACHE_TTL_SECONDS)


def save_current_admin_tool_cache_entry(
        tool_name: str,
        tool_input: Any,
        tool_output: Any,
) -> None:
    """
    功能描述：
        基于当前上下文会话保存管理端工具缓存记录。

    参数说明：
        tool_name (str): 工具名称。
        tool_input (Any): 工具入参。
        tool_output (Any): 工具输出。

    返回值：
        None。

    异常说明：
        ValueError: 当当前上下文未绑定会话 UUID 时抛出。
    """

    conversation_uuid = _get_current_admin_tool_cache_conversation_uuid()
    save_admin_tool_cache_entry(
        conversation_uuid=conversation_uuid,
        tool_name=tool_name,
        tool_input=tool_input,
        tool_output=tool_output,
    )


def render_admin_tool_cache_prompt(conversation_uuid: str) -> str:
    """
    功能描述：
        渲染管理端工具缓存的 prompt 片段。

    参数说明：
        conversation_uuid (str): 当前会话 UUID。

    返回值：
        str: JSON code block 形式的缓存片段；无缓存时返回空字符串。

    异常说明：
        ValueError: 当会话 UUID 为空时抛出。
    """

    cache_payload = load_admin_tool_cache(conversation_uuid)
    if not cache_payload:
        return ""

    sorted_entries = sorted(
        cache_payload.values(),
        key=lambda item: str(item.get("updated_at") or ""),
        reverse=True,
    )
    rendered_payload = {
        "title": ADMIN_TOOL_CACHE_PROMPT_TITLE,
        "records": sorted_entries[:ADMIN_TOOL_CACHE_MAX_PROMPT_RECORDS],
    }
    rendered_json = json.dumps(rendered_payload, ensure_ascii=False, indent=2)
    return f"```json\n{rendered_json}\n```"


__all__ = [
    "ADMIN_TOOL_CACHE_KEY_PREFIX",
    "ADMIN_TOOL_CACHE_MAX_PROMPT_RECORDS",
    "ADMIN_TOOL_CACHE_PROMPT_TITLE",
    "ADMIN_TOOL_CACHE_TTL_SECONDS",
    "AdminToolCacheEntry",
    "bind_current_admin_tool_cache_conversation",
    "build_admin_tool_cache_key",
    "load_admin_tool_cache",
    "render_admin_tool_cache_prompt",
    "reset_current_admin_tool_cache_conversation",
    "save_admin_tool_cache_entry",
    "save_current_admin_tool_cache_entry",
]
