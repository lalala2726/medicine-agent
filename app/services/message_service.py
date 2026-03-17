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
from app.core.database.mongodb import DEFAULT_MESSAGES_COLLECTION, get_mongo_database
from app.core.exception.exceptions import ServiceException
from app.schemas.document.message import (
    MessageCard,
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


def _normalize_content(content: Any) -> str:
    """
    归一化消息内容。

    用途：
    - 统一处理数据库落库前的 content；
    - 将 `None`、非字符串和纯空白内容统一收敛为空字符串；
    - 让上层由 schema 继续判断“该空字符串是否允许落库”。

    Args:
        content: 原始消息内容，允许为任意类型。

    Returns:
        str: 归一化后的消息内容；无有效文本时返回空字符串。
    """

    if not isinstance(content, str):
        return ""
    if not content.strip():
        return ""
    return content


def _normalize_cards(
        cards: list[MessageCard | dict[str, Any]] | None,
) -> list[MessageCard | dict[str, Any]] | None:
    """
    归一化卡片列表。

    用途：
    - 统一处理落库前的 `cards` 字段；
    - 将空列表与 `None` 统一视为“未传卡片”；
    - 保持非空卡片列表的原始顺序不变。

    Args:
        cards: 原始卡片列表，元素可以是 `MessageCard` 或普通字典。

    Returns:
        list[MessageCard | dict[str, Any]] | None:
            非空时返回原列表；为空时返回 `None`。
    """

    if not cards:
        return None
    return cards


def add_message(
        *,
        conversation_id: Annotated[str, Field(min_length=1)],
        role: MessageRole | str,
        status: MessageStatus | str = MessageStatus.SUCCESS,
        content: str,
        thinking: str | None = None,
        token_usage: TokenUsage | dict[str, Any] | None = None,
        cards: list[MessageCard | dict[str, Any]] | None = None,
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
        cards: 可选 AI 卡片列表，仅 ai 消息会保存，user 消息会被忽略。
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
        content=_normalize_content(content),
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
        cards=(
            _normalize_cards(cards)
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
    if document.get("cards") is None:
        document.pop("cards", None)
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


def count_messages(
        *,
        conversation_id: Annotated[str, Field(min_length=1)],
) -> int:
    """
    统计某个会话下的消息总数。

    Args:
        conversation_id: 所属会话 Mongo ObjectId（字符串形式）。

    Returns:
        int: 当前会话命中的消息总数。

    Note:
        仅按 `conversation_id` 统计，不附加其他角色、状态或摘要过滤条件。
    """

    query = {"conversation_id": _to_object_id(conversation_id)}
    db = get_mongo_database()
    collection = db[_resolve_collection_name()]
    return int(collection.count_documents(query))


def _build_summarizable_messages_query(
        *,
        conversation_id: Annotated[str, Field(min_length=1)],
        after_message_id: str | None = None,
) -> dict[str, Any]:
    """
    功能描述：
        构建“可参与摘要”的消息查询条件。

    参数说明：
        conversation_id (str): 会话 Mongo ObjectId（字符串形式）。
        after_message_id (str | None): 可选消息游标，仅查询 `_id` 大于该值的消息。

    返回值：
        dict[str, Any]: MongoDB 查询条件。

    异常说明：
        ServiceException:
            - BAD_REQUEST: `conversation_id` 或 `after_message_id` 不是合法 ObjectId。
    """

    query: dict[str, Any] = {
        "conversation_id": _to_object_id(conversation_id),
        "role": {
            "$in": [
                MessageRole.USER.value,
                MessageRole.AI.value,
            ]
        },
        "status": MessageStatus.SUCCESS.value,
    }
    normalized_after_message_id = (after_message_id or "").strip()
    if normalized_after_message_id:
        try:
            query["_id"] = {"$gt": ObjectId(normalized_after_message_id)}
        except Exception as exc:
            raise ServiceException(
                code=ResponseCode.BAD_REQUEST,
                message="after_message_id 格式不正确",
            ) from exc
    return query


def count_summarizable_messages(
        *,
        conversation_id: Annotated[str, Field(min_length=1)],
        after_message_id: str | None = None,
) -> int:
    """
    功能描述：
        统计某会话中可参与摘要的消息数量。

    参数说明：
        conversation_id (str): 会话 Mongo ObjectId（字符串形式）。
        after_message_id (str | None): 可选消息游标，仅统计 `_id` 更大的消息。

    返回值：
        int: 命中的消息条数。

    异常说明：
        ServiceException:
            - BAD_REQUEST: ID 不是合法 ObjectId。
    """

    query = _build_summarizable_messages_query(
        conversation_id=conversation_id,
        after_message_id=after_message_id,
    )
    db = get_mongo_database()
    collection = db[_resolve_collection_name()]
    return int(collection.count_documents(query))


def list_summarizable_messages(
        *,
        conversation_id: Annotated[str, Field(min_length=1)],
        limit: Annotated[int, Field(ge=1)] = 50,
        after_message_id: str | None = None,
        ascending: bool = True,
) -> list[MessageDocument]:
    """
    功能描述：
        查询某会话中可参与摘要的消息列表。

    参数说明：
        conversation_id (str): 会话 Mongo ObjectId（字符串形式）。
        limit (int): 返回条数上限。
        after_message_id (str | None): 可选消息游标，仅查询 `_id` 更大的消息。
        ascending (bool): 是否按创建时间升序返回；`True` 表示旧到新。

    返回值：
        list[MessageDocument]: 满足摘要条件的消息列表。

    异常说明：
        ServiceException:
            - BAD_REQUEST: ID 不是合法 ObjectId。
    """

    query = _build_summarizable_messages_query(
        conversation_id=conversation_id,
        after_message_id=after_message_id,
    )
    sort_direction = ASCENDING if ascending else DESCENDING
    db = get_mongo_database()
    collection = db[_resolve_collection_name()]
    cursor = collection.find(query).sort("created_at", sort_direction).limit(limit)
    return [_to_message_document(item) for item in cursor]


def list_latest_summarizable_messages(
        *,
        conversation_id: Annotated[str, Field(min_length=1)],
        limit: Annotated[int, Field(ge=1)] = 100,
        after_message_id: str | None = None,
) -> list[MessageDocument]:
    """
    功能描述：
        获取“最新 N 条可参与摘要消息”，并按时间正序返回。

    参数说明：
        conversation_id (str): 会话 Mongo ObjectId（字符串形式）。
        limit (int): 需要的最新消息数量。
        after_message_id (str | None): 可选消息游标，仅查询 `_id` 更大的消息。

    返回值：
        list[MessageDocument]: 最新 N 条消息（旧到新）。

    异常说明：
        ServiceException:
            - BAD_REQUEST: ID 不是合法 ObjectId。
    """

    latest_desc = list_summarizable_messages(
        conversation_id=conversation_id,
        limit=limit,
        after_message_id=after_message_id,
        ascending=False,
    )
    return list(reversed(latest_desc))


def list_summarizable_tail_messages(
        *,
        conversation_id: Annotated[str, Field(min_length=1)],
        limit: Annotated[int, Field(ge=1)] = 20,
) -> list[MessageDocument]:
    """
    功能描述：
        获取会话中可参与摘要的“尾部消息窗口”，并按时间正序返回。

    参数说明：
        conversation_id (str): 会话 Mongo ObjectId（字符串形式）。
        limit (int): 尾部窗口大小。

    返回值：
        list[MessageDocument]: 尾部消息列表（旧到新）。

    异常说明：
        ServiceException:
            - BAD_REQUEST: ID 不是合法 ObjectId。
    """

    return list_latest_summarizable_messages(
        conversation_id=conversation_id,
        limit=limit,
        after_message_id=None,
    )
