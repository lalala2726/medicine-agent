from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any, Callable

from loguru import logger

from app.schemas.sse_response import AssistantResponse

EventPayload = dict[str, Any]
EventContent = dict[str, Any]
EventEmitter = Callable[[EventPayload], None]

_status_emitter: ContextVar[EventEmitter | None] = ContextVar(
    "sse_event_emitter",
    default=None,
)
_current_status_node: ContextVar[str | None] = ContextVar(
    "sse_current_status_node",
    default=None,
)


def set_status_emitter(emitter: EventEmitter | None) -> Token:
    """
    在当前请求上下文中设置 SSE 事件发射器。

    Args:
        emitter: 事件发射函数。传入 `None` 表示清空当前上下文的发射器。
            发射器函数签名应为 `Callable[[dict[str, Any]], None]`。

    Returns:
        Token: ContextVar 的 token，用于后续通过 `reset_status_emitter` 还原上下文。
    """

    return _status_emitter.set(emitter)


def reset_status_emitter(token: Token) -> None:
    """
    根据 token 还原当前请求上下文中的 SSE 事件发射器。

    Args:
        token: 由 `set_status_emitter` 返回的 ContextVar token。

    Returns:
        None: 该函数仅执行上下文恢复操作。
    """

    _status_emitter.reset(token)


def has_status_emitter() -> bool:
    """
    判断当前请求上下文是否已设置 SSE 事件发射器。

    Args:
        无。

    Returns:
        bool: 已设置发射器时返回 `True`，否则返回 `False`。
    """

    return _status_emitter.get() is not None


def get_current_status_node() -> str | None:
    """
    获取当前请求上下文中的“当前状态节点”标识。

    该值用于将工具调用事件自动挂载到父节点（例如在节点执行过程中触发 tool 调用）。

    Args:
        无。

    Returns:
        str | None: 当前节点名；未设置时返回 `None`。
    """

    return _current_status_node.get()


def set_current_status_node(node: str | None) -> Token:
    """
    设置当前请求上下文中的“当前状态节点”标识。

    Args:
        node: 节点名称。传入 `None` 表示清空。

    Returns:
        Token: ContextVar token，用于后续还原。
    """

    return _current_status_node.set(node)


def reset_current_status_node(token: Token) -> None:
    """
    根据 token 还原“当前状态节点”上下文值。

    Args:
        token: 由 `set_current_status_node` 返回的 token。

    Returns:
        None: 该函数仅执行上下文恢复。
    """

    _current_status_node.reset(token)


def _build_event_content(
        *,
        node: str,
        state: str,
        parent_node: str | None = None,
        message: str | None = None,
        result: str | None = None,
        name: str | None = None,
        arguments: str | None = None,
) -> EventContent:
    """
    构建标准 SSE 事件 content 结构。

    Args:
        node: 事件所属节点标识。
        state: 节点状态值（如 `start/end/timely`）。
        parent_node: 父节点标识，常用于工具事件归属。
        message: 给前端展示的提示文案。
        result: 执行结果标记（如 `success/error`）。
        name: 可选名称字段（常用于函数/工具名）。
        arguments: 可选参数字符串（常用于函数调用入参展示）。

    Returns:
        dict[str, Any]: 规范化后的 content 负载字典。
    """

    content: EventContent = {
        "node": node,
        "state": state,
    }
    if parent_node is not None:
        content["parent_node"] = parent_node
    if message is not None:
        content["message"] = message
    if result is not None:
        content["result"] = result
    if name is not None:
        content["name"] = name
    if arguments is not None:
        content["arguments"] = arguments
    return content


def _emit_payload(*, payload: EventPayload, source: str) -> None:
    """
    将原始事件负载发送到当前上下文的 SSE 发射器。

    行为约束：
    - 发射器未设置：记录 warning 并忽略，不中断主流程；
    - 发射器执行异常：记录 warning 并忽略，不中断主流程。

    Args:
        payload: 待发送的事件负载（通常是可序列化字典）。
        source: 事件来源标识，仅用于日志定位。

    Returns:
        None: 该函数只负责尝试发射事件，不返回结果。
    """

    emitter = _status_emitter.get()
    if emitter is None:
        logger.warning(
            "SSE event ignored because emitter is missing: source={}",
            source,
        )
        return

    try:
        emitter(payload)
    except Exception as exc:
        logger.warning(
            "SSE event ignored because emitter failed: source={} error={}",
            source,
            exc,
        )


def _emit_event(*, event_type: str, content: EventContent) -> None:
    """
    发送标准事件信封结构。

    Args:
        event_type: 事件类型字符串（如 `status`、`function_call`）。
        content: 事件内容字典。

    Returns:
        None: 事件通过 `_emit_payload` 推送，不返回值。
    """

    _emit_payload(
        payload={"type": event_type, "content": content},
        source=f"event:{event_type}",
    )


def emit_status(
        *,
        node: str,
        state: str,
        message: str | None = None,
        result: str | None = None,
        name: str | None = None,
        arguments: str | None = None,
) -> None:
    """
    发送节点状态事件（`type=status`）。

    Args:
        node: 节点标识（如 `chat_agent`）。
        state: 状态值（如 `start/end/timely`）。
        message: 可选提示文案。
        result: 可选结果标记（如 `success/error`）。
        name: 可选名称字段。
        arguments: 可选参数字段（字符串形式）。

    Returns:
        None: 事件被推送到 SSE 发射器。
    """

    _emit_event(
        event_type="status",
        content=_build_event_content(
            node=node,
            state=state,
            message=message,
            result=result,
            name=name,
            arguments=arguments,
        ),
    )


def emit_function_call(
        *,
        node: str,
        state: str,
        parent_node: str | None = None,
        message: str | None = None,
        result: str | None = None,
        name: str | None = None,
        arguments: str | None = None,
) -> None:
    """
    发送工具/函数调用事件（`type=function_call`）。

    Args:
        node: 工具节点标识（如 `tool:get_order_list`）。
        state: 状态值（如 `start/end/timely`）。
        parent_node: 父节点标识。若不传，自动读取当前上下文节点。
        message: 可选提示文案。
        result: 可选结果标记（如 `success/error`）。
        name: 可选工具名/函数名。
        arguments: 可选参数字符串。

    Returns:
        None: 事件被推送到 SSE 发射器。
    """

    resolved_parent_node = (
        parent_node if parent_node is not None else _current_status_node.get()
    )
    _emit_event(
        event_type="function_call",
        content=_build_event_content(
            node=node,
            state=state,
            parent_node=resolved_parent_node,
            message=message,
            result=result,
            name=name,
            arguments=arguments,
        ),
    )


def emit_sse_response(response: AssistantResponse) -> None:
    """
    发送任意 `AssistantResponse` 结构事件到 SSE 发射器。

    说明：
    - 该入口会强制 `is_end=False`，确保流结束包由流式引擎统一输出。

    Args:
        response: 业务侧构造的 SSE 响应模型对象。

    Returns:
        None: 事件被序列化后推送到 SSE 发射器。
    """

    normalized_response = response.model_copy(update={"is_end": False})
    payload = normalized_response.model_dump(mode="json", exclude_none=True)
    _emit_payload(payload=payload, source="emit_sse_response")
