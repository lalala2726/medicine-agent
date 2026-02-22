from __future__ import annotations

import datetime
import os
from typing import Annotated

from bson import ObjectId
from pydantic import Field
from pymongo import ReturnDocument
from pymongo.errors import PyMongoError

from app.core.codes import ResponseCode
from app.core.exceptions import ServiceException
from app.core.mongodb import DEFAULT_CONVERSATION_SUMMARIES_COLLECTION, get_mongo_database
from app.schemas.document.conversation_summary import (
    ConversationSummary,
    ConversationSummarySetOnInsert,
    ConversationSummaryUpsertPayload,
    ConversationSummaryUpdateSet,
)


def _resolve_collection_name() -> str:
    """解析 conversation_summaries 集合名，支持环境变量覆盖。"""

    return (
            (os.getenv("MONGODB_CONVERSATION_SUMMARIES_COLLECTION") or DEFAULT_CONVERSATION_SUMMARIES_COLLECTION).strip()
            or DEFAULT_CONVERSATION_SUMMARIES_COLLECTION
    )


def _to_object_id(raw_conversation_id: str) -> ObjectId:
    """将字符串会话ID转换为 MongoDB ObjectId。"""

    try:
        return ObjectId(raw_conversation_id)
    except Exception as exc:  # pragma: no cover - 防御性兜底
        raise ServiceException(
            code=ResponseCode.BAD_REQUEST,
            message="conversation_id 格式不正确",
        ) from exc


def save_conversation_summary(
        *,
        conversation_id: Annotated[str, Field(min_length=1)],
        summary_content: Annotated[str, Field(min_length=1)],
        summarized_messages: list[str] | None = None,
        status: str = "success",
) -> str:
    """保存会话 summary（按 conversation_id upsert）。"""

    normalized_messages = [item for item in (summarized_messages or []) if isinstance(item, str) and item.strip()]
    now = datetime.datetime.now()

    db = get_mongo_database()
    collection = db[_resolve_collection_name()]
    query = {"conversation_id": _to_object_id(conversation_id)}
    update_payload = ConversationSummaryUpsertPayload(
        set_fields=ConversationSummaryUpdateSet(
            summary_content=summary_content,
            summarized_messages=normalized_messages,
            status=status,
            updated_at=now,
        ),
        set_on_insert_fields=ConversationSummarySetOnInsert(
            conversation_id=query["conversation_id"],
            created_at=now,
        ),
    )
    update_doc = update_payload.model_dump(by_alias=True, mode="python")

    try:
        document = collection.find_one_and_update(
            query,
            update_doc,
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
    except PyMongoError as exc:
        raise ServiceException(code=ResponseCode.DATABASE_ERROR, message="数据库错误") from exc

    if not isinstance(document, dict) or "_id" not in document:
        raise ServiceException(code=ResponseCode.DATABASE_ERROR, message="数据库错误")
    return str(document["_id"])


def get_conversation_summary(
        *,
        conversation_id: Annotated[str, Field(min_length=1)],
) -> ConversationSummary | None:
    """按 conversation_id 查询会话 summary。"""

    db = get_mongo_database()
    collection = db[_resolve_collection_name()]
    query = {"conversation_id": _to_object_id(conversation_id)}

    try:
        document = collection.find_one(query)
        if document is None:
            return None
        return ConversationSummary.model_validate(document)
    except PyMongoError as exc:
        raise ServiceException(code=ResponseCode.DATABASE_ERROR, message="数据库错误") from exc
