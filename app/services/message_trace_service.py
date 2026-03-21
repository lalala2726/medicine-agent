from __future__ import annotations

import datetime
from typing import Annotated, Any

from bson import ObjectId
from pydantic import Field

from app.core.codes import ResponseCode
from app.core.database.mongodb import MONGODB_MESSAGE_TRACES_COLLECTION, get_mongo_database
from app.core.exception.exceptions import ServiceException
from app.core.llms.provider import LlmProvider, resolve_provider
from app.schemas.document.message_trace import (
    ExecutionTraceItem,
    MessageTraceCreate,
    MessageTraceDocument,
    MessageTraceProvider,
    TraceTokenUsage,
    WorkflowTraceSummary,
)

_GATEWAY_NODE_NAME = "gateway_router"
_WORKFLOW_NAME = "admin_assistant_graph"
_ALLOWED_WORKFLOW_STATUSES = {"success", "error", "cancelled"}
"""message_trace 支持的工作流终态集合。"""


def _resolve_collection_name() -> str:
    """
    功能描述：
        返回 `message_traces` 集合固定名称常量。

    参数说明：
        无。

    返回值：
        str: 最终使用的 MongoDB 集合名称。

    异常说明：
        无。
    """

    return MONGODB_MESSAGE_TRACES_COLLECTION


def _to_object_id(raw_conversation_id: str) -> ObjectId:
    """
    功能描述：
        将字符串会话 ID 转换为 MongoDB `ObjectId`，用于标准化落库字段类型。

    参数说明：
        raw_conversation_id (str): 原始会话 ID 字符串。

    返回值：
        ObjectId: 可用于 Mongo 查询/写入的会话 ID。

    异常说明：
        ServiceException:
            - 当会话 ID 不是合法 ObjectId 时抛出，错误码为 `BAD_REQUEST`。
    """

    try:
        return ObjectId(raw_conversation_id)
    except Exception as exc:  # pragma: no cover - 防御性兜底
        raise ServiceException(
            code=ResponseCode.BAD_REQUEST,
            message="conversation_id 格式不正确",
        ) from exc


def _sort_execution_trace(
        execution_trace: list[ExecutionTraceItem],
) -> list[ExecutionTraceItem]:
    """
    功能描述：
        按 `sequence` 对节点执行轨迹做稳定排序，确保执行路径可预期。

    参数说明：
        execution_trace (list[ExecutionTraceItem]): 归一化后的节点执行轨迹列表。

    返回值：
        list[ExecutionTraceItem]:
            按 `sequence` 升序排序后的轨迹列表；`sequence` 相同场景保持原有顺序。

    异常说明：
        无。
    """

    indexed_trace = list(enumerate(execution_trace))
    indexed_trace.sort(key=lambda item: (item[1].sequence, item[0]))
    return [item[1] for item in indexed_trace]


def _normalize_execution_trace(
        execution_trace: list[ExecutionTraceItem | dict[str, Any]] | None,
) -> list[ExecutionTraceItem]:
    """
    功能描述：
        归一化 `execution_trace`，自动过滤非法项并返回稳定顺序结果。

    参数说明：
        execution_trace (list[ExecutionTraceItem | dict[str, Any]] | None):
            原始节点执行轨迹；支持模型对象列表或字典列表，允许为空。

    返回值：
        list[ExecutionTraceItem]:
            解析成功的节点轨迹列表；当输入为空或全部非法时返回空列表。

    异常说明：
        无；单条轨迹解析失败会被忽略，不影响其余条目。
    """

    if execution_trace is None:
        return []

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
    return _sort_execution_trace(normalized_items)


def _normalize_trace_token_usage(
        token_usage: TraceTokenUsage | dict[str, Any] | None,
) -> TraceTokenUsage | None:
    """
    功能描述：
        归一化 trace 专用 token 汇总结构。

    参数说明：
        token_usage (TraceTokenUsage | dict[str, Any] | None):
            trace token 汇总；支持模型对象、字典或空值。

    返回值：
        TraceTokenUsage | None:
            合法结构返回标准模型；不合法或为空时返回 `None`。

    异常说明：
        无；校验失败时直接返回 `None`。
    """

    if token_usage is None:
        return None
    if isinstance(token_usage, TraceTokenUsage):
        return token_usage
    if not isinstance(token_usage, dict):
        return None

    try:
        return TraceTokenUsage.model_validate(token_usage)
    except Exception:
        return None


def _normalize_route_targets(raw_route_targets: Any) -> list[str]:
    """
    功能描述：
        标准化 gateway 路由目标数组并做顺序去重。

    参数说明：
        raw_route_targets (Any): gateway 节点上下文中的原始 route_targets 字段。

    返回值：
        list[str]:
            清洗后的目标节点数组；字段不存在或无有效值时返回空列表。

    异常说明：
        无。
    """

    if not isinstance(raw_route_targets, list):
        return []

    normalized_targets: list[str] = []
    for raw_target in raw_route_targets:
        target = str(raw_target or "").strip()
        if not target:
            continue
        if target in normalized_targets:
            continue
        normalized_targets.append(target)
    return normalized_targets


def _resolve_task_difficulty(raw_task_difficulty: Any) -> str | None:
    """
    功能描述：
        标准化任务难度字段，仅允许 `normal/high`。

    参数说明：
        raw_task_difficulty (Any): 原始任务难度字段。

    返回值：
        str | None:
            合法难度值返回 `normal` 或 `high`；其他场景返回 `None`。

    异常说明：
        无。
    """

    task_difficulty = str(raw_task_difficulty or "").strip()
    if task_difficulty in {"normal", "high"}:
        return task_difficulty
    return None


def _build_workflow_summary(
        *,
        execution_trace: list[ExecutionTraceItem],
        workflow_status: str,
        workflow_name: str,
) -> WorkflowTraceSummary:
    """
    功能描述：
        从节点执行轨迹构建工作流级追踪汇总。

    参数说明：
        execution_trace (list[ExecutionTraceItem]): 归一化后的节点轨迹列表。
        workflow_status (str): 工作流终态，支持 `success/error/cancelled`。
        workflow_name (str): 工作流名称。

    返回值：
        WorkflowTraceSummary:
            包含 `workflow_status/execution_path/final_node/route_targets/task_difficulty`
            的工作流汇总对象。

    异常说明：
        无。
    """

    execution_path = [item.node_name for item in execution_trace]
    final_node = execution_path[-1] if execution_path else None

    gateway_context: dict[str, Any] = {}
    for trace_item in execution_trace:
        if trace_item.node_name != _GATEWAY_NODE_NAME:
            continue
        if isinstance(trace_item.node_context, dict):
            gateway_context = trace_item.node_context
        break

    route_targets = _normalize_route_targets(gateway_context.get("route_targets"))
    task_difficulty = _resolve_task_difficulty(gateway_context.get("task_difficulty"))

    return WorkflowTraceSummary(
        workflow_name=workflow_name,
        workflow_status=workflow_status if workflow_status in _ALLOWED_WORKFLOW_STATUSES else "success",
        execution_path=execution_path,
        final_node=final_node,
        route_targets=route_targets,
        task_difficulty=task_difficulty,
    )


def _resolve_message_trace_provider(
        provider: MessageTraceProvider | LlmProvider | str | None = None,
) -> MessageTraceProvider:
    """
    功能描述：
        解析 message_trace 落库时使用的模型厂家，统一遵循
        “显式参数 > 环境配置（LLM_PROVIDER）> openai” 的规则。

    参数说明：
        provider (MessageTraceProvider | LlmProvider | str | None):
            调用方显式传入的厂家；默认值 `None`。
            支持 `openai/aliyun/volcengine` 及 `LlmProvider.ALIYUN` 风格字符串。

    返回值：
        MessageTraceProvider: 归一化后的厂家枚举值。

    异常说明：
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
        workflow_name: str = _WORKFLOW_NAME,
        execution_trace: list[ExecutionTraceItem | dict[str, Any]] | None = None,
        token_usage: TraceTokenUsage | dict[str, Any] | None = None,
        workflow_status: str | None = None,
        has_error: bool = False,
) -> str | None:
    """
    功能描述：
        新增一条 message_trace 记录，始终落库 workflow 汇总与节点追踪。

    参数说明：
        message_uuid (str): 消息 UUID。
        conversation_id (str): 会话 ObjectId 字符串。
        provider (MessageTraceProvider | LlmProvider | str | None):
            模型厂家；未传时按环境默认解析。
        workflow_name (str): 工作流名称，默认 `admin_assistant_graph`。
        execution_trace (list[ExecutionTraceItem | dict[str, Any]] | None):
            节点轨迹列表；为空时按空列表落库。
        token_usage (TraceTokenUsage | dict[str, Any] | None):
            trace token 汇总；为空时不写该字段。
        workflow_status (str | None): 显式工作流终态；为空时由 `has_error` 推导。
        has_error (bool): 流程是否发生错误，用于生成 workflow_status。

    返回值：
        str | None:
            插入成功时返回插入文档 ID；当底层驱动未返回 ID 时返回 `None`。

    异常说明：
        ServiceException:
            - `conversation_id` 非法时抛出 `BAD_REQUEST`。
        ValueError:
            - `provider` 非法时由 `_resolve_message_trace_provider` 抛出。
    """

    normalized_execution_trace = _normalize_execution_trace(execution_trace)
    normalized_token_usage = _normalize_trace_token_usage(token_usage)
    normalized_provider = _resolve_message_trace_provider(provider)
    resolved_workflow_status = (
            str(workflow_status or "").strip().lower()
            or ("error" if has_error else "success")
    )
    workflow_summary = _build_workflow_summary(
        execution_trace=normalized_execution_trace,
        workflow_status=resolved_workflow_status,
        workflow_name=workflow_name,
    )

    payload = MessageTraceCreate(
        message_uuid=message_uuid,
        conversation_id=conversation_id,
        provider=normalized_provider,
        workflow=workflow_summary,
        execution_trace=normalized_execution_trace,
        token_usage=normalized_token_usage,
    )

    now = datetime.datetime.now()
    document = payload.model_dump(mode="python", exclude_none=True)
    document["conversation_id"] = _to_object_id(payload.conversation_id)
    document["created_at"] = now
    document["updated_at"] = now

    db = get_mongo_database()
    collection = db[_resolve_collection_name()]
    result = collection.insert_one(document)
    inserted_id = getattr(result, "inserted_id", None)
    return None if inserted_id is None else str(inserted_id)


def get_message_trace_by_message_uuid(
        *,
        message_uuid: Annotated[str, Field(min_length=1)],
) -> MessageTraceDocument | None:
    """
    功能描述：
        按 message_uuid 查询单条 message_trace 文档并返回强类型模型。

    参数说明：
        message_uuid (str): 消息 UUID。

    返回值：
        MessageTraceDocument | None:
            命中时返回文档模型，未命中时返回 `None`。

    异常说明：
        无；数据库异常由全局异常处理链路统一处理。
    """

    db = get_mongo_database()
    collection = db[_resolve_collection_name()]
    document = collection.find_one({"message_uuid": message_uuid})
    if document is None:
        return None
    return MessageTraceDocument.model_validate(document)
