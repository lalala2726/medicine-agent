from __future__ import annotations

import json
import os
from typing import Any, Awaitable, Callable, Mapping, TypeVar

from langchain.agents.middleware import AgentMiddleware
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.types import Command
from loguru import logger
from app.core.agent.agent_event_bus import emit_function_call

_DEFAULT_TOOL_TIMELY_MESSAGE = "工具正在持续处理中，请稍后查看结果"
_TOOL_STATUS_META_ATTR = "__tool_call_status_config__"
_REGISTERED_TOOL_STATUS_MESSAGES: dict[str, dict[str, str]] = {}
_TOOL_RESULT_SUCCESS_MESSAGE = "工具调用成功"
_ToolLike = TypeVar("_ToolLike")


def _default_start_message(display_name: str) -> str:
    """生成默认开始文案。"""

    return f"正在调用工具 {display_name}"


def _default_error_message(display_name: str) -> str:
    """生成默认错误文案。"""

    return f"工具 {display_name} 调用失败"


def _is_tool_log_enabled() -> bool:
    """
    判断是否开启工具调用日志。

    Returns:
        bool: 环境变量 `AGENT_TOOL_LOG_ENABLED` 开启时返回 True。
    """

    value = os.getenv("AGENT_TOOL_LOG_ENABLED", "false").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _resolve_tool_name(tool_obj: Any) -> str:
    """
    解析工具注册名。

    Args:
        tool_obj: 被 `@tool_call_status` 装饰的对象（函数或 LangChain Tool）。

    Returns:
        str: 工具在 LangChain 中可被调用的名称。

    Raises:
        ValueError: 当无法解析工具名称时抛出。
    """

    tool_name = getattr(tool_obj, "name", None)
    if isinstance(tool_name, str) and tool_name.strip():
        return tool_name.strip()

    callable_name = getattr(tool_obj, "__name__", None)
    if isinstance(callable_name, str) and callable_name.strip():
        return callable_name.strip()

    raise ValueError("无法解析工具名称，请在可调用函数或 LangChain Tool 上使用 @tool_call_status。")


def _extract_error_message_from_content(content: Any) -> str | None:
    """
    从工具结果内容中提取错误信息。

    Args:
        content: 工具返回内容。

    Returns:
        str | None: 提取到错误信息时返回错误文本，否则返回 None。
    """

    if isinstance(content, Mapping):
        error_value = content.get("error")
        if error_value:
            return str(error_value)
        return None

    if isinstance(content, list):
        for item in content:
            message = _extract_error_message_from_content(item)
            if message:
                return message
        return None

    text = str(content or "").strip()
    if not text:
        return None

    if "__ERROR__:" in text:
        return text.split("__ERROR__:", 1)[1].strip() or text

    try:
        payload = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None

    if isinstance(payload, Mapping):
        error_value = payload.get("error")
        if error_value:
            return str(error_value)
    return None


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

    config = _REGISTERED_TOOL_STATUS_MESSAGES.get(tool_name, {})
    display_name = str(config.get("display_name") or tool_name)
    start_message = str(config.get("start") or _default_start_message(display_name))
    error_message = str(config.get("error") or _default_error_message(display_name))
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
    工具状态装饰器：登记工具状态文案

    Args:
        tool_name: 工具显示名。未传时使用工具函数名。
        start_message: 工具开始文案。
        error_message: 工具失败文案。
        timely_message: 工具持续处理文案。

    Returns:
        Callable: 装饰器函数，返回原对象（不包裹执行逻辑）。
    """

    def _decorate(tool_obj: _ToolLike) -> _ToolLike:
        resolved_tool_name = _resolve_tool_name(tool_obj)
        display_name = tool_name or resolved_tool_name

        # 如果未指定文本，则使用默认文本
        resolved_start = start_message if start_message is not None else _default_start_message(display_name)
        resolved_error = error_message if error_message is not None else _default_error_message(display_name)
        resolved_timely = timely_message if timely_message is not None else _DEFAULT_TOOL_TIMELY_MESSAGE

        config = {
            "display_name": display_name,
            "start": resolved_start,
            "error": resolved_error,
            "timely": resolved_timely,
        }
        _REGISTERED_TOOL_STATUS_MESSAGES[resolved_tool_name] = config
        try:
            setattr(tool_obj, _TOOL_STATUS_META_ATTR, config)
        except Exception:
            # 部分第三方对象可能不允许动态写属性，不影响中间件按注册表工作。
            pass
        return tool_obj

    return _decorate


def build_tool_status_middleware(*, enabled: bool = True):
    """
    构建工具状态发送中间件。

    仅对被 `@tool_call_status` 标注过的工具发送状态事件。

    Args:
        enabled: 是否启用工具状态发送。

    Returns:
        AgentMiddleware: 可直接传入 `create_agent(..., middleware=[...])` 的中间件实例。
    """

    class _ToolStatusMiddleware(AgentMiddleware):
        """
        工具状态发送中间件（支持同步/异步工具调用链路）。
        """

        def __init__(self, *, enabled_flag: bool) -> None:
            super().__init__()
            self.enabled = enabled_flag

        @staticmethod
        def _parse_request(request: ToolCallRequest) -> tuple[str, str, Any]:
            """
            从 ToolCallRequest 提取工具调用核心信息。

            Args:
                request: 当前工具调用请求。

            Returns:
                tuple[str, str, Any]:
                    - 工具名称；
                    - 工具调用 ID；
                    - 工具参数。
            """

            tool_call = request.tool_call if isinstance(request.tool_call, Mapping) else {}
            tool_name = str(tool_call.get("name") or "").strip()
            tool_call_id = str(tool_call.get("id") or "tool_call").strip() or "tool_call"
            tool_args = tool_call.get("args")
            return tool_name, tool_call_id, tool_args

        @staticmethod
        def _format_arguments(tool_args: Any) -> str:
            """
            格式化工具参数用于事件透传。

            Args:
                tool_args: 工具参数。

            Returns:
                str: 序列化后的参数文本。
            """

            if isinstance(tool_args, (dict, list)):
                return json.dumps(tool_args, ensure_ascii=False, default=str)
            return str(tool_args)

        @staticmethod
        def _to_error_tool_message(*, error_message: str, tool_call_id: str) -> ToolMessage:
            """
            构造工具失败兜底 ToolMessage。

            Args:
                error_message: 失败文案。
                tool_call_id: 工具调用 ID。

            Returns:
                ToolMessage: 返回给模型的错误工具消息。
            """

            return ToolMessage(
                content=json.dumps({"error": error_message}, ensure_ascii=False),
                tool_call_id=tool_call_id,
            )

        @staticmethod
        def _finalize_result(
            *,
            tool_name: str,
            tool_call_id: str,
            tool_args: Any,
            timely_message: str,
            result: ToolMessage | Command | None,
            log_enabled: bool,
        ) -> ToolMessage | Command | None:
            """
            对工具返回结果做统一收尾处理。

            Args:
                tool_name: 工具名称。
                tool_call_id: 工具调用 ID。
                tool_args: 工具参数。
                timely_message: 工具持续执行文案。
                result: 工具返回结果。
                log_enabled: 是否开启工具日志。

            Returns:
                ToolMessage | Command | None: 原始结果或转换后的结果。
            """

            if result is None:
                emit_function_call(
                    node=f"tool:{tool_name}",
                    state="timely",
                    message=timely_message,
                    name=tool_name,
                )
                return result

            resolved_result_error: str | None = None
            if isinstance(result, ToolMessage):
                resolved_result_error = _extract_error_message_from_content(result.content)

            if resolved_result_error:
                logger.error(
                    "Tool call returned error payload. tool_name={} tool_call_id={} error={} args={}",
                    tool_name,
                    tool_call_id,
                    resolved_result_error,
                    tool_args,
                )
            elif log_enabled:
                logger.info(
                    "Tool call success. tool_name={} tool_call_id={}",
                    tool_name,
                    tool_call_id,
                )

            emit_function_call(
                node=f"tool:{tool_name}",
                state="end",
                result="error" if resolved_result_error else "success",
                message=resolved_result_error or _TOOL_RESULT_SUCCESS_MESSAGE,
                name=tool_name,
            )
            return result

        @staticmethod
        def _handle_failure(
            *,
            tool_name: str,
            tool_call_id: str,
            tool_args: Any,
            error_message: str,
            exc: Exception,
        ) -> ToolMessage:
            """
            统一处理工具调用异常并生成错误 ToolMessage。

            Args:
                tool_name: 工具名称。
                tool_call_id: 工具调用 ID。
                tool_args: 工具参数。
                error_message: 输出给用户侧的错误文案。
                exc: 原始异常对象。

            Returns:
                ToolMessage: 返回给模型的错误工具消息。
            """

            logger.opt(exception=exc).error(
                "Tool call failed. tool_name={} tool_call_id={} args={}",
                tool_name,
                tool_call_id,
                tool_args,
            )
            emit_function_call(
                node=f"tool:{tool_name}",
                state="end",
                result="error",
                message=error_message,
                name=tool_name,
            )
            return _ToolStatusMiddleware._to_error_tool_message(
                error_message=error_message,
                tool_call_id=tool_call_id,
            )

        def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCallRequest], ToolMessage | Command | None],
        ) -> ToolMessage | Command | None:
            """
            同步工具调用拦截。

            Args:
                request: 工具调用请求。
                handler: 下一个工具处理器（同步）。

            Returns:
                ToolMessage | Command | None: 工具执行结果。
            """

            log_enabled = _is_tool_log_enabled()
            if not self.enabled:
                return handler(request)

            tool_name, tool_call_id, tool_args = self._parse_request(request)
            if not tool_name or tool_name not in _REGISTERED_TOOL_STATUS_MESSAGES:
                return handler(request)

            start_message, error_message, timely_message = resolve_tool_call_messages(tool_name)
            if log_enabled:
                logger.info(
                    "Tool call start. tool_name={} tool_call_id={} args={}",
                    tool_name,
                    tool_call_id,
                    tool_args,
                )
            emit_function_call(
                node=f"tool:{tool_name}",
                state="start",
                message=start_message,
                name=tool_name,
                arguments=self._format_arguments(tool_args),
            )

            try:
                result = handler(request)
            except Exception as exc:
                return self._handle_failure(
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    tool_args=tool_args,
                    error_message=error_message or str(exc),
                    exc=exc,
                )

            return self._finalize_result(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                tool_args=tool_args,
                timely_message=timely_message,
                result=result,
                log_enabled=log_enabled,
            )

        async def awrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command | None]],
        ) -> ToolMessage | Command | None:
            """
            异步工具调用拦截。

            Args:
                request: 工具调用请求。
                handler: 下一个工具处理器（异步）。

            Returns:
                ToolMessage | Command | None: 工具执行结果。
            """

            log_enabled = _is_tool_log_enabled()
            if not self.enabled:
                return await handler(request)

            tool_name, tool_call_id, tool_args = self._parse_request(request)
            if not tool_name or tool_name not in _REGISTERED_TOOL_STATUS_MESSAGES:
                return await handler(request)

            start_message, error_message, timely_message = resolve_tool_call_messages(tool_name)
            if log_enabled:
                logger.info(
                    "Tool call start. tool_name={} tool_call_id={} args={}",
                    tool_name,
                    tool_call_id,
                    tool_args,
                )
            emit_function_call(
                node=f"tool:{tool_name}",
                state="start",
                message=start_message,
                name=tool_name,
                arguments=self._format_arguments(tool_args),
            )

            try:
                result = await handler(request)
            except Exception as exc:
                return self._handle_failure(
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    tool_args=tool_args,
                    error_message=error_message or str(exc),
                    exc=exc,
                )

            return self._finalize_result(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                tool_args=tool_args,
                timely_message=timely_message,
                result=result,
                log_enabled=log_enabled,
            )

    return _ToolStatusMiddleware(enabled_flag=enabled)
