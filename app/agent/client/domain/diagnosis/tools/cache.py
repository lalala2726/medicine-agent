from __future__ import annotations

import json
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from app.core.database.redis import RedisHashCache

# 诊断工具缓存 Redis key 前缀。
DIAGNOSIS_TOOL_CACHE_KEY_PREFIX = "diagnosis:tool_cache"
# 诊断工具缓存 TTL，单位秒。
DIAGNOSIS_TOOL_CACHE_TTL_SECONDS = 1800
# 症状候选检索工具缓存字段名。
TOOL_CACHE_FIELD_SEARCH_SYMPTOM_CANDIDATES = "search_symptom_candidates"
# 候选疾病召回工具缓存字段名。
TOOL_CACHE_FIELD_QUERY_DISEASE_CANDIDATES = "query_disease_candidates_by_symptoms"
# 单疾病详情查询工具缓存字段名。
TOOL_CACHE_FIELD_QUERY_DISEASE_DETAIL = "query_disease_detail"
# 批量疾病详情查询工具缓存字段名。
TOOL_CACHE_FIELD_QUERY_DISEASE_DETAILS = "query_disease_details"
# 追问症状候选工具缓存字段名。
TOOL_CACHE_FIELD_QUERY_FOLLOWUP_SYMPTOMS = "query_followup_symptom_candidates"
# 诊断工具缓存字段固定顺序。
DIAGNOSIS_TOOL_CACHE_FIELDS = (
    TOOL_CACHE_FIELD_SEARCH_SYMPTOM_CANDIDATES,
    TOOL_CACHE_FIELD_QUERY_DISEASE_CANDIDATES,
    TOOL_CACHE_FIELD_QUERY_DISEASE_DETAIL,
    TOOL_CACHE_FIELD_QUERY_DISEASE_DETAILS,
    TOOL_CACHE_FIELD_QUERY_FOLLOWUP_SYMPTOMS,
)
# 诊断工具缓存提示词固定标题。
DIAGNOSIS_TOOL_CACHE_PROMPT_TITLE = "已缓存诊断工具结果"

_DIAGNOSIS_TOOL_CACHE_CONVERSATION_UUID: ContextVar[str | None] = ContextVar(
    "diagnosis_tool_cache_conversation_uuid",
    default=None,
)


def _normalize_required_text(value: str, *, field_name: str) -> str:
    """规范化必填字符串。

    Args:
        value: 原始字符串值。
        field_name: 当前字段名称。

    Returns:
        str: 去除首尾空白后的非空字符串。

    Raises:
        ValueError: 归一化后为空时抛出。
    """

    normalized_value = str(value or "").strip()
    if not normalized_value:
        raise ValueError(f"{field_name} 不能为空")
    return normalized_value


def _serialize_cache_value(value: Any) -> Any:
    """将缓存值转换为可 JSON 序列化结构。

    Args:
        value: 任意待缓存对象。

    Returns:
        Any: 可直接进行 JSON 序列化的结构。
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
    """将 Redis 原始值解码为文本。

    Args:
        value: Redis 原始返回值。
        field_name: 当前字段名称。

    Returns:
        str: 解码后的文本内容。
    """

    if isinstance(value, bytes):
        return value.decode("utf-8")
    return _normalize_required_text(str(value or ""), field_name=field_name)


def build_diagnosis_tool_cache_key(conversation_uuid: str) -> str:
    """构造诊断工具缓存 Redis key。

    Args:
        conversation_uuid: 当前会话 UUID。

    Returns:
        str: 诊断工具缓存 Redis key。
    """

    normalized_uuid = _normalize_required_text(
        conversation_uuid,
        field_name="conversation_uuid",
    )
    return f"{DIAGNOSIS_TOOL_CACHE_KEY_PREFIX}:{normalized_uuid}"


def bind_current_diagnosis_tool_cache_conversation(conversation_uuid: str) -> Token[str | None]:
    """绑定当前诊断工具缓存使用的会话 UUID。

    Args:
        conversation_uuid: 当前会话 UUID。

    Returns:
        Token[str | None]: `ContextVar` 重置令牌。
    """

    normalized_uuid = _normalize_required_text(
        conversation_uuid,
        field_name="conversation_uuid",
    )
    return _DIAGNOSIS_TOOL_CACHE_CONVERSATION_UUID.set(normalized_uuid)


def reset_current_diagnosis_tool_cache_conversation(token: Token[str | None]) -> None:
    """重置当前诊断工具缓存绑定的会话 UUID。

    Args:
        token: 绑定时返回的 `ContextVar` 重置令牌。

    Returns:
        None
    """

    _DIAGNOSIS_TOOL_CACHE_CONVERSATION_UUID.reset(token)


def _get_current_diagnosis_tool_cache_conversation_uuid() -> str:
    """读取当前诊断工具缓存绑定的会话 UUID。

    Returns:
        str: 当前会话 UUID。

    Raises:
        ValueError: 当前上下文未绑定会话 UUID 时抛出。
    """

    conversation_uuid = _DIAGNOSIS_TOOL_CACHE_CONVERSATION_UUID.get()
    return _normalize_required_text(
        str(conversation_uuid or ""),
        field_name="conversation_uuid",
    )


def load_diagnosis_tool_cache(conversation_uuid: str) -> dict[str, Any]:
    """读取当前会话的诊断工具缓存。

    Args:
        conversation_uuid: 当前会话 UUID。

    Returns:
        dict[str, Any]: 按固定字段顺序组织的诊断工具缓存映射。
    """

    cache_key = build_diagnosis_tool_cache_key(conversation_uuid)
    raw_hash = RedisHashCache().h_get_all(cache_key)
    if not raw_hash:
        return {}

    decoded_payload: dict[str, Any] = {}
    for raw_field, raw_value in raw_hash.items():
        decoded_field = _decode_redis_text(raw_field, field_name="cache_field")
        decoded_payload[decoded_field] = json.loads(
            _decode_redis_text(raw_value, field_name=decoded_field),
        )

    ordered_payload: dict[str, Any] = {}
    for field_name in DIAGNOSIS_TOOL_CACHE_FIELDS:
        if field_name in decoded_payload:
            ordered_payload[field_name] = decoded_payload[field_name]
    return ordered_payload


def save_diagnosis_tool_cache_entry(
        conversation_uuid: str,
        tool_name: str,
        tool_input: Any,
        tool_output: Any,
) -> None:
    """保存单条诊断工具缓存记录。

    Args:
        conversation_uuid: 当前会话 UUID。
        tool_name: 诊断工具名称，同时作为 Redis Hash 字段名。
        tool_input: 工具入参。
        tool_output: 工具成功返回结果。

    Returns:
        None
    """

    cache_key = build_diagnosis_tool_cache_key(conversation_uuid)
    normalized_tool_name = _normalize_required_text(tool_name, field_name="tool_name")
    cache_entry = {
        "tool_name": normalized_tool_name,
        "tool_input": _serialize_cache_value(tool_input),
        "tool_output": _serialize_cache_value(tool_output),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    redis_hash_cache = RedisHashCache()
    redis_hash_cache.h_put(
        cache_key,
        normalized_tool_name,
        json.dumps(cache_entry, ensure_ascii=False),
    )
    redis_hash_cache.expire(cache_key, DIAGNOSIS_TOOL_CACHE_TTL_SECONDS)


def save_current_diagnosis_tool_cache_entry(
        tool_name: str,
        tool_input: Any,
        tool_output: Any,
) -> None:
    """按当前上下文会话保存诊断工具缓存记录。

    Args:
        tool_name: 诊断工具名称，同时作为 Redis Hash 字段名。
        tool_input: 工具入参。
        tool_output: 工具成功返回结果。

    Returns:
        None
    """

    conversation_uuid = _get_current_diagnosis_tool_cache_conversation_uuid()
    save_diagnosis_tool_cache_entry(
        conversation_uuid,
        tool_name,
        tool_input,
        tool_output,
    )


def render_diagnosis_tool_cache_prompt(conversation_uuid: str) -> str:
    """渲染诊断工具缓存提示词段落。

    Args:
        conversation_uuid: 当前会话 UUID。

    Returns:
        str: 诊断工具缓存提示词；无缓存时返回空字符串。
    """

    cache_payload = load_diagnosis_tool_cache(conversation_uuid)
    if not cache_payload:
        return ""
    rendered_json = json.dumps(cache_payload, ensure_ascii=False, indent=2)
    return f"```json\n{rendered_json}\n```"


__all__ = [
    "DIAGNOSIS_TOOL_CACHE_FIELDS",
    "DIAGNOSIS_TOOL_CACHE_KEY_PREFIX",
    "DIAGNOSIS_TOOL_CACHE_PROMPT_TITLE",
    "DIAGNOSIS_TOOL_CACHE_TTL_SECONDS",
    "TOOL_CACHE_FIELD_QUERY_DISEASE_CANDIDATES",
    "TOOL_CACHE_FIELD_QUERY_DISEASE_DETAIL",
    "TOOL_CACHE_FIELD_QUERY_DISEASE_DETAILS",
    "TOOL_CACHE_FIELD_QUERY_FOLLOWUP_SYMPTOMS",
    "TOOL_CACHE_FIELD_SEARCH_SYMPTOM_CANDIDATES",
    "bind_current_diagnosis_tool_cache_conversation",
    "build_diagnosis_tool_cache_key",
    "load_diagnosis_tool_cache",
    "render_diagnosis_tool_cache_prompt",
    "reset_current_diagnosis_tool_cache_conversation",
    "save_current_diagnosis_tool_cache_entry",
    "save_diagnosis_tool_cache_entry",
]
