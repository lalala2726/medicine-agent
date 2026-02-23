from __future__ import annotations

import inspect
from functools import wraps
from typing import Callable

from app.core.sse_event_bus import emit_function_call

_TOOL_FUNCTION_CALL_MESSAGES: dict[str, dict[str, str]] = {
    "get_order_list": {
        "start": "正在查询订单信息",
        "error": "订单服务调用失败",
        "timely": "订单信息正在持续处理中",
    },
    "get_supported_chart_types": {
        "start": "正在获取系统支持的图表类型",
        "error": "获取图表类型失败",
        "timely": "图表类型正在持续处理中",
    },
    "get_chart_sample_by_name": {
        "start": "正在获取图表配置模板",
        "error": "获取图表模板失败",
    },
}

_DEFAULT_TOOL_TIMELY_MESSAGE = "工具正在持续处理中，请稍后查看结果"


def resolve_tool_call_messages(tool_name: str) -> tuple[str, str, str]:
    """
    根据工具名解析工具调用阶段文案（start/error/timely）。

    Args:
        tool_name: 工具名（通常为函数名或业务定义的工具标识）。

    Returns:
        tuple[str, str, str]:
            - start_message: 工具开始执行提示文案；
            - error_message: 工具失败提示文案；
            - timely_message: 工具持续处理中提示文案。
    """

    config = _TOOL_FUNCTION_CALL_MESSAGES.get(tool_name, {})
    start_message = config.get("start") or f"正在调用工具 {tool_name}"
    error_message = config.get("error") or f"工具 {tool_name} 调用失败"
    timely_message = config.get("timely") or _DEFAULT_TOOL_TIMELY_MESSAGE
    return start_message, error_message, timely_message


def tool_call_status(
        *,
        tool_name: str | None = None,
        start_message: str | None = None,
        error_message: str | None = None,
        timely_message: str | None = None,
) -> Callable:
    """
    工具状态装饰器：自动发送 `function_call` 的 start/end/error/timely 事件。

    Args:
        tool_name: 工具名称。未传时默认使用被装饰函数名。
        start_message: 自定义开始文案；未传则使用映射默认值。
        error_message: 自定义错误文案；未传则使用映射默认值。
        timely_message: 自定义持续中提示；未传则使用映射默认值。

    Returns:
        Callable: 装饰器函数。可装饰同步或异步工具函数。
    """

    def _decorate(func: Callable) -> Callable:
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

