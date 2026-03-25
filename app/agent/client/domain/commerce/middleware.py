"""
客户端 commerce 动态工具注入中间件。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse

from app.agent.client.domain.commerce.registry import ClientCommerceToolRegistry


class CommerceDynamicToolMiddleware(AgentMiddleware):
    """
    功能描述：
        根据 state 中的 `loaded_tool_keys` 动态过滤 commerce 管理工具。

    参数说明：
        registry (ClientCommerceToolRegistry): 工具注册中心。

    返回值：
        无（中间件对象）。

    异常说明：
        无。
    """

    def __init__(self, *, registry: ClientCommerceToolRegistry) -> None:
        """
        功能描述：
            初始化动态工具中间件。

        参数说明：
            registry (ClientCommerceToolRegistry): 工具注册中心。

        返回值：
            None。

        异常说明：
            无。
        """

        self._registry = registry

    @staticmethod
    def _normalize_loaded_tool_keys(state: Any) -> list[str] | None:
        """
        功能描述：
            从请求状态中读取并规范化已加载工具数组。

        参数说明：
            state (Any): 请求状态对象。

        返回值：
            list[str] | None: 合法的工具 key 数组；未命中时返回 `None`。

        异常说明：
            无。
        """

        if not isinstance(state, Mapping):
            return None

        raw_loaded_tool_keys = state.get("loaded_tool_keys")
        if not isinstance(raw_loaded_tool_keys, list):
            return None

        normalized_tool_keys: list[str] = []
        for raw_tool_key in raw_loaded_tool_keys:
            tool_key = str(raw_tool_key or "").strip()
            if not tool_key:
                continue
            if tool_key in normalized_tool_keys:
                continue
            normalized_tool_keys.append(tool_key)
        return normalized_tool_keys

    def _filter_request_tools(self, request: ModelRequest) -> ModelRequest:
        """
        功能描述：
            过滤当前模型请求中的工具列表。

        参数说明：
            request (ModelRequest): 当前模型请求对象。

        返回值：
            ModelRequest: 已应用动态工具过滤后的请求对象。

        异常说明：
            无。
        """

        request_tools = list(request.tools)
        state_dict = request.state if isinstance(request.state, Mapping) else {}
        loaded_tool_keys = self._normalize_loaded_tool_keys(state_dict)
        visible_tools = self._registry.filter_visible_tools(
            request_tools=request_tools,
            loaded_tool_keys=loaded_tool_keys,
        )
        return request.override(tools=visible_tools)

    def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """
        功能描述：
            在同步模型调用前执行工具过滤。

        参数说明：
            request (ModelRequest): 当前模型请求对象。
            handler (Callable[[ModelRequest], ModelResponse]): 下游处理器。

        返回值：
            ModelResponse: 下游模型响应。

        异常说明：
            无。
        """

        filtered_request = self._filter_request_tools(request)
        return handler(filtered_request)

    async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """
        功能描述：
            在异步模型调用前执行工具过滤。

        参数说明：
            request (ModelRequest): 当前模型请求对象。
            handler (Callable[[ModelRequest], Awaitable[ModelResponse]]): 下游处理器。

        返回值：
            ModelResponse: 下游模型响应。

        异常说明：
            无。
        """

        filtered_request = self._filter_request_tools(request)
        return await handler(filtered_request)


__all__ = [
    "CommerceDynamicToolMiddleware",
]
