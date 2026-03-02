from __future__ import annotations

import datetime
import os
import uuid
from typing import Annotated, Any, Mapping

from bson import ObjectId
from loguru import logger
from pydantic import Field
from pymongo import ASCENDING, DESCENDING

from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException
from app.core.mongodb import DEFAULT_MESSAGES_COLLECTION, get_mongo_database
from app.schemas.document.message import (
    MessageRole,
    MessageCreate,
    MessageDocument,
    MessageStatus,
    TokenUsage,
)


def _resolve_collection_name() -> str:
    """解析 messages 集合名，支持环境变量覆盖。"""

    return (
            (os.getenv("MONGODB_MESSAGES_COLLECTION") or DEFAULT_MESSAGES_COLLECTION).strip()
            or DEFAULT_MESSAGES_COLLECTION
    )


def _to_object_id(raw_conversation_id: str) -> ObjectId:
    """
    将字符串会话ID转换为 MongoDB ObjectId。

    数据表约束要求 `conversation_id` 为 `objectId`，这里统一做转换与错误拦截。
    """

    try:
        return ObjectId(raw_conversation_id)
    except Exception as exc:  # pragma: no cover - 防御性兜底
        raise ServiceException(
            code=ResponseCode.BAD_REQUEST,
            message="conversation_id 格式不正确",
        ) from exc


def _to_message_document(document: dict[str, Any]) -> MessageDocument:
    """
    将 Mongo 原始文档转换为消息模型。

    统一由 schema 做字段归一化（如 ObjectId -> str），避免业务层重复转换逻辑。
    """

    return MessageDocument.model_validate(document)


def _to_non_negative_int(value: Any) -> int | None:
    """将任意值转换为非负整数。"""

    if value is None:
        return None
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return None
    if resolved < 0:
        return None
    return resolved


def _normalize_token_usage(
        token_usage: TokenUsage | dict[str, Any] | None,
) -> TokenUsage | None:
    """归一化 token_usage，提取 prompt/completion/total 总量字段。"""

    if token_usage is None:
        return None
    if isinstance(token_usage, TokenUsage):
        return token_usage
    if not isinstance(token_usage, Mapping):
        return None

    prompt_tokens = _to_non_negative_int(token_usage.get("prompt_tokens"))
    completion_tokens = _to_non_negative_int(token_usage.get("completion_tokens"))
    total_tokens = _to_non_negative_int(token_usage.get("total_tokens"))
    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return None

    resolved_prompt = prompt_tokens or 0
    resolved_completion = completion_tokens or 0
    resolved_total = total_tokens
    if resolved_total is None:
        resolved_total = resolved_prompt + resolved_completion

    try:
        return TokenUsage(
            prompt_tokens=resolved_prompt,
            completion_tokens=resolved_completion,
            total_tokens=resolved_total,
        )
    except Exception:
        logger.warning(
            "Ignore invalid token_usage while persisting message. payload={payload}",
            payload=token_usage,
        )
        return None


def _normalize_thinking(thinking: Any) -> str | None:
    """归一化 thinking 文本，空值或空白内容返回 None。"""

    if not isinstance(thinking, str):
        return None
    normalized = thinking.strip()
    return normalized or None


def add_message(
        *,
        conversation_id: Annotated[str, Field(min_length=1)],
        role: MessageRole | str,
        status: MessageStatus | str = MessageStatus.SUCCESS,
        content: Annotated[str, Field(min_length=1)],
        thinking: str | None = None,
        token_usage: TokenUsage | dict[str, Any] | None = None,
        message_uuid: str | None = None,
) -> str:
    """
    新增一条会话消息。

    Args:
        conversation_id: 所属会话 Mongo ObjectId（字符串形式）。
        role: 消息角色（user/ai）。
        status: 消息状态（success/error）。
        content: 消息内容。
        thinking: 可选 AI 深度思考完整文本，仅 ai 消息会保存。
        token_usage: 可选 token 使用总量，仅支持
            prompt_tokens/completion_tokens/total_tokens 三个字段。
            且仅 ai 消息会保存，user 消息会被忽略。
        message_uuid: 可选消息 UUID，不传时自动生成。

    Returns:
        str: 新增消息的 Mongo ObjectId 字符串。

    Raises:
        ServiceException: 当参数不合法时抛出。

    Note:
        数据库异常会由全局异常处理器统一拦截。
    """

    normalized_role = MessageRole(role)
    payload = MessageCreate(
        uuid=message_uuid or str(uuid.uuid4()),
        conversation_id=conversation_id,
        role=normalized_role,
        status=status,
        content=content,
        thinking=(
            _normalize_thinking(thinking)
            if normalized_role == MessageRole.AI
            else None
        ),
        token_usage=(
            _normalize_token_usage(token_usage)
            if normalized_role == MessageRole.AI
            else None
        ),
    )

    now = datetime.datetime.now()
    # Mongo 写入文档统一由 Pydantic 模型序列化产出。
    document = payload.model_dump()
    if document.get("thinking") is None:
        document.pop("thinking", None)
    if document.get("token_usage") is None:
        document.pop("token_usage", None)
    document["conversation_id"] = _to_object_id(payload.conversation_id)
    document["created_at"] = now
    document["updated_at"] = now

    db = get_mongo_database()
    collection = db[_resolve_collection_name()]
    result = collection.insert_one(document)
    return str(result.inserted_id)


def get_message_by_uuid(message_uuid: Annotated[str, Field(min_length=1)]) -> MessageDocument | None:
    """
    按消息 UUID 查询单条会话消息。

    Args:
        message_uuid: 消息业务唯一ID。

    Returns:
        MessageDocument | None: 命中返回消息模型，否则返回 None。

    Note:
        数据库异常会由全局异常处理器统一拦截。
    """

    db = get_mongo_database()
    collection = db[_resolve_collection_name()]
    document = collection.find_one({"uuid": message_uuid})
    if document is None:
        return None
    return _to_message_document(document)


def list_messages(
        *,
        conversation_id: Annotated[str, Field(min_length=1)],
        limit: Annotated[int, Field(ge=1)] = 50,
        skip: Annotated[int, Field(ge=0)] = 0,
        ascending: bool = True,
) -> list[MessageDocument]:
    """
    查询某个会话下的消息列表。

    Args:
        conversation_id: 所属会话 Mongo ObjectId（字符串形式）。
        limit: 返回条数上限，默认 50。
        skip: 跳过条数，默认 0。
        ascending: 是否按创建时间升序，默认 True（旧到新）。

    Returns:
        list[MessageDocument]: 消息文档模型列表。
            仅负责数据库模型输出，不做上层会话消息格式转换。

    Note:
        数据库异常会由全局异常处理器统一拦截。
    """

    sort_direction = ASCENDING if ascending else DESCENDING
    query = {"conversation_id": _to_object_id(conversation_id)}

    db = get_mongo_database()
    collection = db[_resolve_collection_name()]
    cursor = collection.find(query).sort("created_at", sort_direction).skip(skip).limit(limit)
    return [_to_message_document(item) for item in cursor]
