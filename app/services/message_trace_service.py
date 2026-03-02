from __future__ import annotations

import datetime
import os
from typing import Annotated, Any

from bson import ObjectId
from pydantic import Field

from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException
from app.core.llms.provider import LlmProvider, resolve_provider
from app.core.mongodb import DEFAULT_MESSAGE_TRACES_COLLECTION, get_mongo_database
from app.schemas.document.message_trace import (
    ExecutionTraceItem,
    MessageTraceCreate,
    MessageTraceDocument,
    MessageTraceProvider,
    MessageTraceTokenDetail,
)


def _resolve_collection_name() -> str:
    """解析 message_traces 集合名，支持环境变量覆盖。"""

    return (
            (os.getenv("MONGODB_MESSAGE_TRACES_COLLECTION") or DEFAULT_MESSAGE_TRACES_COLLECTION).strip()
            or DEFAULT_MESSAGE_TRACES_COLLECTION
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


def _normalize_execution_trace(
        execution_trace: list[ExecutionTraceItem | dict[str, Any]] | None,
) -> list[ExecutionTraceItem] | None:
    """归一化 execution_trace，自动忽略非法项。"""

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
            continue
    return normalized_items or None


def _normalize_token_usage_detail(
        token_usage_detail: MessageTraceTokenDetail | dict[str, Any] | None,
) -> MessageTraceTokenDetail | None:
    """归一化 token_usage_detail，严格按 MessageTraceTokenDetail 校验。"""

    if token_usage_detail is None:
        return None
    if isinstance(token_usage_detail, MessageTraceTokenDetail):
        return token_usage_detail
    if not isinstance(token_usage_detail, dict):
        return None

    try:
        return MessageTraceTokenDetail.model_validate(token_usage_detail)
    except Exception:
        return None


def _resolve_message_trace_provider(
        provider: MessageTraceProvider | LlmProvider | str | None = None,
) -> MessageTraceProvider:
    """
    功能描述:
        解析 message_trace 落库时使用的模型厂家，统一遵循
        “显式参数 > 环境配置（LLM_PROVIDER）> openai” 的规则。

    参数说明:
        provider (MessageTraceProvider | LlmProvider | str | None):
            调用方显式传入的厂家；默认值 `None`。
            支持 `openai/aliyun/volcengine` 及 `LlmProvider.ALIYUN` 风格字符串。

    返回值:
        MessageTraceProvider: 归一化后的厂家枚举值。

    异常说明:
        ValueError:
            当 provider 取值不在支持范围时，由 `resolve_provider` 抛出。
    """

    resolved = resolve_provider(provider)
    return MessageTraceProvider(resolved.value)


def add_message_trace(
        *,
        message_uuid: Annotated[str, Field(min_length=1)],
        conversation_id: Annotated[str, Field(min_length=1)],
        provider: MessageTraceProvider | LlmProvider | str | None = None,
        execution_trace: list[ExecutionTraceItem | dict[str, Any]] | None = None,
        token_usage_detail: MessageTraceTokenDetail | dict[str, Any] | None = None,
) -> str | None:
    """
    新增一条 message_trace。

    当 execution_trace 与 token_usage_detail 都为空时，直接跳过写入并返回 None。

    Note:
        数据库异常会由全局异常处理器统一拦截。
    """

    normalized_execution_trace = _normalize_execution_trace(execution_trace)
    normalized_token_usage_detail = _normalize_token_usage_detail(token_usage_detail)
    if normalized_execution_trace is None and normalized_token_usage_detail is None:
        return None

    normalized_provider = _resolve_message_trace_provider(provider)
    payload = MessageTraceCreate(
        message_uuid=message_uuid,
        conversation_id=conversation_id,
        provider=normalized_provider,
        execution_trace=normalized_execution_trace,
        token_usage_detail=normalized_token_usage_detail,
    )

    now = datetime.datetime.now()
    # Mongo 写入文档统一由 Pydantic 模型序列化产出。
    document = payload.model_dump()
    if document.get("execution_trace") is None:
        document.pop("execution_trace", None)
    if document.get("token_usage_detail") is None:
        document.pop("token_usage_detail", None)
    document["conversation_id"] = _to_object_id(payload.conversation_id)
    document["created_at"] = now
    document["updated_at"] = now

    db = get_mongo_database()
    collection = db[_resolve_collection_name()]
    result = collection.insert_one(document)
    return str(result.inserted_id)


def get_message_trace_by_message_uuid(
        *,
        message_uuid: Annotated[str, Field(min_length=1)],
) -> MessageTraceDocument | None:
    """
    按 message_uuid 查询单条 message_trace。

    Note:
        数据库异常会由全局异常处理器统一拦截。
    """

    db = get_mongo_database()
    collection = db[_resolve_collection_name()]
    document = collection.find_one({"message_uuid": message_uuid})
    if document is None:
        return None
    return MessageTraceDocument.model_validate(document)
