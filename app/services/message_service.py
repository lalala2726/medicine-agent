from __future__ import annotations

import datetime
import os
import uuid
from typing import Annotated, Any

from bson import ObjectId
from langchain_core.messages import AIMessage, HumanMessage
from loguru import logger
from pydantic import Field
from pymongo import ASCENDING, DESCENDING
from pymongo.errors import PyMongoError

from app.core.codes import ResponseCode
from app.core.exceptions import ServiceException
from app.core.mongodb import DEFAULT_ADMIN_MESSAGES_COLLECTION, get_mongo_database
from app.schemas.admin_message import (
    AdminMessageCreate,
    AdminMessageDocument,
    ExecutionTraceItem,
    MessageRole,
    MessageStatus,
    TokenUsage,
)


def _resolve_collection_name() -> str:
    """解析 admin_messages 集合名，支持环境变量覆盖。"""

    return (
        (os.getenv("MONGODB_ADMIN_MESSAGES_COLLECTION") or DEFAULT_ADMIN_MESSAGES_COLLECTION).strip()
        or DEFAULT_ADMIN_MESSAGES_COLLECTION
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


def _to_message_document(document: dict[str, Any]) -> AdminMessageDocument:
    """
    将 Mongo 原始文档转换为消息模型。

    统一由 schema 做字段归一化（如 ObjectId -> str），避免业务层重复转换逻辑。
    """

    return AdminMessageDocument.model_validate(document)


def _normalize_execution_trace(
        execution_trace: list[ExecutionTraceItem | dict[str, Any]] | None,
) -> list[ExecutionTraceItem] | None:
    """
    归一化 execution_trace，自动忽略非法项。

    Args:
        execution_trace: 原始节点执行追踪数据。

    Returns:
        list[ExecutionTraceItem] | None: 归一化结果，无有效项时返回 None。
    """

    if execution_trace is None:
        return None

    normalized_items: list[ExecutionTraceItem] = []
    for item in execution_trace:
        if isinstance(item, ExecutionTraceItem):
            normalized_items.append(item)
            continue
        if not isinstance(item, dict):
            continue
        try:
            normalized_items.append(ExecutionTraceItem.model_validate(item))
        except Exception:
            logger.warning("Ignore invalid execution_trace item while persisting message.")
    return normalized_items or None


def _normalize_token_usage(
        token_usage: TokenUsage | dict[str, Any] | None,
) -> TokenUsage | None:
    """归一化 token_usage，自动忽略非法数据。"""

    if token_usage is None:
        return None
    if isinstance(token_usage, TokenUsage):
        return token_usage
    if not isinstance(token_usage, dict):
        return None
    try:
        return TokenUsage.model_validate(token_usage)
    except Exception:
        logger.warning("Ignore invalid token_usage while persisting message.")
        return None


def add_message(
        *,
        conversation_id: Annotated[str, Field(min_length=1)],
        role: MessageRole | str,
        status: MessageStatus | str = MessageStatus.SUCCESS,
        content: Annotated[str, Field(min_length=1)],
        thought_chain: list[Any] | None = None,
        token_usage: TokenUsage | dict[str, Any] | None = None,
        execution_trace: list[ExecutionTraceItem | dict[str, Any]] | None = None,
        message_uuid: str | None = None,
) -> str:
    """
    新增一条管理助手消息。

    Args:
        conversation_id: 所属会话 Mongo ObjectId（字符串形式）。
        role: 消息角色（user/assistant）。
        status: 消息状态（success/error）。
        content: 消息内容。
        thought_chain: 可选思维链结构。
        token_usage: 可选 token 使用明细。通常来源于 workflow state 的 `token_usage`，
            且仅 assistant 消息会保存，user 消息会被忽略。
        execution_trace: 可选节点执行追踪明细。通常来源于 workflow state 的
            `execution_traces`，用于复盘本轮节点/工具调用路径。
        message_uuid: 可选消息 UUID，不传时自动生成。

    Returns:
        str: 新增消息的 Mongo ObjectId 字符串。

    Raises:
        ServiceException: 当参数不合法或数据库操作失败时抛出。
    """

    normalized_role = MessageRole(role)
    payload = AdminMessageCreate(
        uuid=message_uuid or str(uuid.uuid4()),
        conversation_id=conversation_id,
        role=normalized_role,
        status=status,
        content=content,
        thought_chain=thought_chain,
        token_usage=(
            _normalize_token_usage(token_usage)
            if normalized_role == MessageRole.ASSISTANT
            else None
        ),
        execution_trace=_normalize_execution_trace(execution_trace),
    )

    now = datetime.datetime.now()
    document = payload.model_dump()
    if document.get("token_usage") is None:
        document.pop("token_usage", None)
    document["conversation_id"] = _to_object_id(payload.conversation_id)
    document["created_at"] = now
    document["updated_at"] = now

    db = get_mongo_database()
    collection = db[_resolve_collection_name()]
    try:
        result = collection.insert_one(document)
        return str(result.inserted_id)
    except PyMongoError as exc:
        raise ServiceException(code=ResponseCode.DATABASE_ERROR, message="数据库错误") from exc


def get_message_by_uuid(
        *,
        message_uuid: Annotated[str, Field(min_length=1)],
) -> AdminMessageDocument | None:
    """
    按消息 UUID 查询单条管理助手消息。

    Args:
        message_uuid: 消息业务唯一ID。

    Returns:
        AdminMessageDocument | None: 命中返回消息模型，否则返回 None。
    """

    db = get_mongo_database()
    collection = db[_resolve_collection_name()]
    try:
        document = collection.find_one({"uuid": message_uuid})
        if document is None:
            return None
        return _to_message_document(document)
    except PyMongoError as exc:
        raise ServiceException(code=ResponseCode.DATABASE_ERROR, message="数据库错误") from exc


def list_messages(
        *,
        conversation_id: Annotated[str, Field(min_length=1)],
        limit: Annotated[int, Field(ge=1)] = 50,
        skip: Annotated[int, Field(ge=0)] = 0,
        ascending: bool = True,
) -> list[AdminMessageDocument]:
    """
    查询某个会话下的消息列表。

    Args:
        conversation_id: 所属会话 Mongo ObjectId（字符串形式）。
        limit: 返回条数上限，默认 50。
        skip: 跳过条数，默认 0。
        ascending: 是否按创建时间升序，默认 True（旧到新）。

    Returns:
        list[AdminMessageDocument]: 消息模型列表。
    """

    sort_direction = ASCENDING if ascending else DESCENDING
    query = {"conversation_id": _to_object_id(conversation_id)}

    db = get_mongo_database()
    collection = db[_resolve_collection_name()]
    try:
        cursor = collection.find(query).sort("created_at", sort_direction).skip(skip).limit(limit)
        return [_to_message_document(item) for item in cursor]
    except PyMongoError as exc:
        raise ServiceException(code=ResponseCode.DATABASE_ERROR, message="数据库错误") from exc


def _to_langchain_message(
        message: AdminMessageDocument,
) -> HumanMessage | AIMessage:
    """
    将消息文档映射为 LangChain 消息对象。

    映射规则：
    - role=user -> HumanMessage
    - role=assistant -> AIMessage
    """

    if message.role == MessageRole.USER:
        return HumanMessage(content=message.content)
    return AIMessage(content=message.content)


def get_history(
        *,
        conversation_id: Annotated[str, Field(min_length=1)],
        limit: Annotated[int, Field(ge=1)] = 50,
        ascending: bool = True,
) -> list[HumanMessage | AIMessage]:
    """
    查询并构建对话历史（LangChain 消息格式）。

    Args:
        conversation_id: 所属会话 Mongo ObjectId（字符串形式）。
        limit: 返回条数上限，默认 50。
        ascending: 是否按创建时间升序，默认 True（旧到新）。

    Returns:
        list[HumanMessage | AIMessage]: 可直接喂给模型的历史消息列表。
    """

    message_list = list_messages(
        conversation_id=conversation_id,
        limit=limit,
        ascending=ascending,
    )
    return [_to_langchain_message(item) for item in message_list]
