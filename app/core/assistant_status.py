from __future__ import annotations

import inspect
from contextvars import ContextVar, Token
from functools import wraps
from typing import Any, Callable, Literal

EventPayload = dict[str, Any]
EventContent = dict[str, Any]
EventEmitter = Callable[[EventPayload], None]
StatusDisplayWhen = Literal["always", "after_coordinator", "never"]

_status_emitter: ContextVar[EventEmitter | None] = ContextVar(
    "assistant_status_emitter",
    default=None,
)

_TOOL_FUNCTION_CALL_MESSAGES: dict[str, dict[str, str]] = {
    "get_order_list": {
        "start": "正在查询订单信息",
        "error": "订单服务调用失败",
        "timely": "订单信息正在持续处理中",
    },
    "get_chart_sample_by_name": {
        "start": "正在获取图表配置模板",
        "error": "获取图表模板失败",
    },
}

_DEFAULT_TOOL_TIMELY_MESSAGE = "工具正在持续处理中，请稍后查看结果"


def set_status_emitter(emitter: EventEmitter | None) -> Token:
    """在当前请求上下文设置状态事件发射器。"""
    return _status_emitter.set(emitter)


def reset_status_emitter(token: Token) -> None:
    """重置状态事件发射器，避免跨请求污染。"""
    _status_emitter.reset(token)


def _build_event_content(
        *,
        node: str,
        state: str,
        message: str | None = None,
        result: str | None = None,
        name: str | None = None,
        arguments: str | None = None,
) -> EventContent:
    content: EventContent = {
        "node": node,
        "state": state,
    }
    if message is not None:
        content["message"] = message
    if result is not None:
        content["result"] = result
    if name is not None:
        content["name"] = name
    if arguments is not None:
        content["arguments"] = arguments
    return content


def _emit_event(*, event_type: str, content: EventContent) -> None:
    """发射事件信封；未注册发射器时静默忽略。"""
    emitter = _status_emitter.get()
    if emitter is None:
        return

    try:
        emitter({"type": event_type, "content": content})
    except Exception:
        # 状态事件不应影响主流程
        return


def emit_status(
        *,
        node: str,
        state: str,
        message: str | None = None,
        result: str | None = None,
        name: str | None = None,
        arguments: str | None = None,
) -> None:
    """发射节点状态事件（type=status）。"""
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
        message: str | None = None,
        result: str | None = None,
        name: str | None = None,
        arguments: str | None = None,
) -> None:
    """发射工具调用事件（type=function_call）。"""
    _emit_event(
        event_type="function_call",
        content=_build_event_content(
            node=node,
            state=state,
            message=message,
            result=result,
            name=name,
            arguments=arguments,
        ),
    )


def resolve_tool_call_messages(tool_name: str) -> tuple[str, str, str]:
    """返回工具调用的 start / error / timely 文案。"""
    config = _TOOL_FUNCTION_CALL_MESSAGES.get(tool_name, {})
    start_message = config.get("start") or f"正在调用工具 {tool_name}"
    error_message = config.get("error") or f"工具 {tool_name} 调用失败"
    timely_message = config.get("timely") or _DEFAULT_TOOL_TIMELY_MESSAGE
    return start_message, error_message, timely_message


def _extract_state_from_call(
        args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any] | None:
    """从装饰器调用参数中提取状态字典。"""
    state = kwargs.get("state")
    if isinstance(state, dict):
        return state

    if args and isinstance(args[0], dict):
        return args[0]

    return None


def _should_emit_status(
        display_when: StatusDisplayWhen,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
) -> bool:
    """判断本次调用是否需要发状态事件。"""
    if display_when == "always":
        return True
    if display_when == "never":
        return False

    state = _extract_state_from_call(args, kwargs)
    if not isinstance(state, dict):
        return False

    routing = state.get("routing")
    if not isinstance(routing, dict):
        return False

    return routing.get("route_target") == "coordinator_agent"


def status_node(
        *,
        node: str,
        start_message: str,
        error_message: str | None = None,
        display_when: StatusDisplayWhen = "always",
):
    """
    节点状态装饰器：自动发 start/end/error 状态事件。

    Args:
        node: 状态事件中的节点名。
        start_message: 节点开始执行时的提示文案。
        error_message: 节点失败时的兜底错误文案。
        display_when: 状态显示策略。
            - always: 始终发状态事件。
            - after_coordinator: 仅当 state.routing.route_target == "coordinator_agent" 时发事件。
              若 route_target 缺失或非法，则默认不发。
            - never: 永不发状态事件。
    """

    def _decorate(func):
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def _async_wrapper(*args, **kwargs):
                should_emit = _should_emit_status(display_when, args, kwargs)
                if should_emit:
                    emit_status(node=node, state="start", message=start_message)
                try:
                    result = await func(*args, **kwargs)
                except Exception as exc:
                    if should_emit:
                        emit_status(
                            node=node,
                            state="end",
                            result="error",
                            message=error_message or str(exc),
                        )
                    raise
                if should_emit:
                    emit_status(node=node, state="end")
                return result

            return _async_wrapper

        @wraps(func)
        def _wrapper(*args, **kwargs):
            should_emit = _should_emit_status(display_when, args, kwargs)
            if should_emit:
                emit_status(node=node, state="start", message=start_message)
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                if should_emit:
                    emit_status(
                        node=node,
                        state="end",
                        result="error",
                        message=error_message or str(exc),
                    )
                raise
            if should_emit:
                emit_status(node=node, state="end")
            return result

        return _wrapper

    return _decorate


def tool_call_status(
        *,
        tool_name: str | None = None,
        start_message: str | None = None,
        error_message: str | None = None,
        timely_message: str | None = None,
):
    """工具状态装饰器：自动发 function_call start/end/error/timely。"""

    def _decorate(func):
        resolved_tool_name = tool_name or func.__name__
        mapped_start, mapped_error, mapped_timely = resolve_tool_call_messages(
            resolved_tool_name
        )
        resolved_start = start_message if start_message is not None else mapped_start
        resolved_error = error_message if error_message is not None else mapped_error
        resolved_timely = (
            timely_message if timely_message is not None else mapped_timely
        )
        node = f"tool:{resolved_tool_name}"

        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def _async_wrapper(*args, **kwargs):
                emit_function_call(
                    node=node,
                    state="start",
                    message=resolved_start,
                )
                try:
                    result = await func(*args, **kwargs)
                except Exception as exc:
                    emit_function_call(
                        node=node,
                        state="end",
                        result="error",
                        message=resolved_error or str(exc),
                    )
                    raise

                if result is None:
                    emit_function_call(
                        node=node,
                        state="timely",
                        message=resolved_timely,
                    )
                    return result

                emit_function_call(node=node, state="end")
                return result

            return _async_wrapper

        @wraps(func)
        def _wrapper(*args, **kwargs):
            emit_function_call(
                node=node,
                state="start",
                message=resolved_start,
            )
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                emit_function_call(
                    node=node,
                    state="end",
                    result="error",
                    message=resolved_error or str(exc),
                )
                raise

            if result is None:
                emit_function_call(
                    node=node,
                    state="timely",
                    message=resolved_timely,
                )
                return result

            emit_function_call(node=node, state="end")
            return result

        return _wrapper

    return _decorate
