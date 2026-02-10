from __future__ import annotations

import inspect
from contextvars import ContextVar, Token
from functools import wraps
from typing import Any, Callable

StatusPayload = dict[str, Any]
StatusEmitter = Callable[[StatusPayload], None]

_status_emitter: ContextVar[StatusEmitter | None] = ContextVar(
    "assistant_status_emitter",
    default=None,
)

_TOOL_STATUS_MESSAGES: dict[str, dict[str, str]] = {
    "get_order_list": {
        "start": "正在查询订单信息",
        "error": "订单服务调用失败",
    },
}


def set_status_emitter(emitter: StatusEmitter | None) -> Token:
    """在当前请求上下文设置状态事件发射器。"""
    return _status_emitter.set(emitter)


def reset_status_emitter(token: Token) -> None:
    """重置状态事件发射器，避免跨请求污染。"""
    _status_emitter.reset(token)


def emit_status(
    *,
    node: str,
    state: str,
    message: str | None = None,
    result: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> None:
    """发射节点/工具状态事件；未注册发射器时静默忽略。"""
    emitter = _status_emitter.get()
    if emitter is None:
        return

    payload: StatusPayload = {
        "node": node,
        "state": state,
    }
    if message is not None:
        payload["message"] = message
    if result is not None:
        payload["result"] = result
    if name is not None:
        payload["name"] = name
    if arguments is not None:
        payload["arguments"] = arguments

    try:
        emitter(payload)
    except Exception:
        # 状态事件不应影响主流程
        return


def resolve_tool_status_messages(tool_name: str) -> tuple[str, str]:
    """返回工具调用的 start / error 文案。"""
    config = _TOOL_STATUS_MESSAGES.get(tool_name, {})
    start_message = config.get("start") or f"正在调用工具 {tool_name}"
    error_message = config.get("error") or f"工具 {tool_name} 调用失败"
    return start_message, error_message


def status_node(
    *,
    node: str,
    start_message: str,
    error_message: str | None = None,
):
    """节点状态装饰器：自动发 start/end/error 状态事件。"""

    def _decorate(func):
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def _async_wrapper(*args, **kwargs):
                emit_status(node=node, state="start", message=start_message)
                try:
                    result = await func(*args, **kwargs)
                except Exception as exc:
                    emit_status(
                        node=node,
                        state="end",
                        result="error",
                        message=error_message or str(exc),
                    )
                    raise
                emit_status(node=node, state="end")
                return result

            return _async_wrapper

        @wraps(func)
        def _wrapper(*args, **kwargs):
            emit_status(node=node, state="start", message=start_message)
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                emit_status(
                    node=node,
                    state="end",
                    result="error",
                    message=error_message or str(exc),
                )
                raise
            emit_status(node=node, state="end")
            return result

        return _wrapper

    return _decorate
