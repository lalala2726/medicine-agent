"""
客户端 commerce 工具注册中心。
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from app.agent.client.domain.commerce.after_sale import (
    check_after_sale_eligibility,
    get_after_sale_detail,
    open_user_after_sale_list,
)
from app.agent.client.domain.commerce.base import (
    create_list_loadable_tools_tool,
    create_load_tools_tool,
)
from app.agent.client.domain.commerce.order import (
    check_order_cancelable,
    get_order_detail,
    get_order_shipping,
    get_order_timeline,
    open_user_order_list,
)
from app.agent.client.domain.commerce.product import (
    get_product_detail,
    get_product_spec,
    search_products,
)


class ClientCommerceToolRegistry:
    """
    功能描述：
        统一维护 client commerce 的基础工具、业务工具和工具索引。

    参数说明：
        无。

    返回值：
        无（注册中心对象）。

    异常说明：
        ValueError: 当工具 key 重复或缺失时抛出。
    """

    def __init__(self) -> None:
        """
        功能描述：
            初始化工具注册中心并构建工具索引。

        参数说明：
            无。

        返回值：
            None。

        异常说明：
            ValueError: 当工具 key 重复或非法时抛出。
        """

        self._business_tools_by_domain: dict[str, tuple[BaseTool, ...]] = {
            "order": (
                open_user_order_list,
                get_order_detail,
                get_order_shipping,
                get_order_timeline,
                check_order_cancelable,
            ),
            "product": (
                search_products,
                get_product_detail,
                get_product_spec,
            ),
            "after_sale": (
                open_user_after_sale_list,
                get_after_sale_detail,
                check_after_sale_eligibility,
            ),
        }
        self._business_tools: tuple[BaseTool, ...] = tuple(
            tool_obj
            for domain_tools in self._business_tools_by_domain.values()
            for tool_obj in domain_tools
        )
        self._list_loadable_tools = create_list_loadable_tools_tool(
            get_tool_catalog=self.get_business_tool_catalog,
        )
        self._load_tools = create_load_tools_tool(
            get_allowed_tool_keys=self.get_business_tool_key_set,
        )
        self._base_tools: tuple[BaseTool, ...] = (
            self._list_loadable_tools,
            self._load_tools,
        )
        self._managed_tools: tuple[BaseTool, ...] = (
            *self._base_tools,
            *self._business_tools,
        )
        self._tool_by_key = self._build_tool_index(self._managed_tools)

    @staticmethod
    def _build_tool_index(tools: tuple[BaseTool, ...]) -> dict[str, BaseTool]:
        """
        功能描述：
            根据工具对象数组构建 `tool_key -> tool` 索引。

        参数说明：
            tools (tuple[BaseTool, ...]): 参与注册的工具数组。

        返回值：
            dict[str, BaseTool]: 工具索引字典。

        异常说明：
            ValueError: 当工具 key 缺失或重复时抛出。
        """

        tool_index: dict[str, BaseTool] = {}
        for tool_obj in tools:
            tool_key = str(getattr(tool_obj, "name", "") or "").strip()
            if not tool_key:
                raise ValueError("工具缺少 name，无法注册")
            if tool_key in tool_index:
                raise ValueError(f"工具 key 重复：{tool_key}")
            tool_index[tool_key] = tool_obj
        return tool_index

    @property
    def all_tools(self) -> list[BaseTool]:
        """
        功能描述：
            返回 `create_agent` 使用的全量已注册工具列表。

        参数说明：
            无。

        返回值：
            list[BaseTool]: 全量工具列表。

        异常说明：
            无。
        """

        return list(self._managed_tools)

    @property
    def base_tools(self) -> list[BaseTool]:
        """
        功能描述：
            返回默认直接暴露给模型的基础工具列表。

        参数说明：
            无。

        返回值：
            list[BaseTool]: 基础工具列表。

        异常说明：
            无。
        """

        return list(self._base_tools)

    def get_business_tool_key_set(self) -> set[str]:
        """
        功能描述：
            返回业务工具 key 集合。

        参数说明：
            无。

        返回值：
            set[str]: 业务工具 key 集合。

        异常说明：
            无。
        """

        return {
            str(tool_obj.name).strip()
            for tool_obj in self._business_tools
        }

    def get_business_tool_catalog(self) -> dict[str, list[str]]:
        """
        功能描述：
            返回按领域分组的业务工具目录。

        参数说明：
            无。

        返回值：
            dict[str, list[str]]: 按领域分组的业务工具精确名称映射。

        异常说明：
            无。
        """

        business_tool_catalog: dict[str, list[str]] = {}
        for domain_name, domain_tools in self._business_tools_by_domain.items():
            business_tool_catalog[domain_name] = [
                str(tool_obj.name).strip()
                for tool_obj in domain_tools
                if str(tool_obj.name).strip()
            ]
        return business_tool_catalog

    def get_base_tool_key_set(self) -> set[str]:
        """
        功能描述：
            返回基础工具 key 集合。

        参数说明：
            无。

        返回值：
            set[str]: 基础工具 key 集合。

        异常说明：
            无。
        """

        return {
            str(tool_obj.name).strip()
            for tool_obj in self._base_tools
        }

    def get_managed_tool_key_set(self) -> set[str]:
        """
        功能描述：
            返回注册中心管理的全部工具 key 集合。

        参数说明：
            无。

        返回值：
            set[str]: 全量工具 key 集合。

        异常说明：
            无。
        """

        return set(self._tool_by_key.keys())

    def resolve_visible_tool_key_set(self, loaded_tool_keys: list[str] | None) -> set[str]:
        """
        功能描述：
            计算当前模型可见的管理工具 key 集合。

        参数说明：
            loaded_tool_keys (list[str] | None): 当前状态中的已加载工具 key 数组。

        返回值：
            set[str]: 当前应暴露给模型的管理工具 key 集合。

        异常说明：
            无。
        """

        visible_tool_keys = self.get_base_tool_key_set()
        business_tool_keys = self.get_business_tool_key_set()
        if not loaded_tool_keys:
            return visible_tool_keys

        for raw_tool_key in loaded_tool_keys:
            tool_key = str(raw_tool_key or "").strip()
            if not tool_key:
                continue
            if tool_key not in business_tool_keys:
                continue
            visible_tool_keys.add(tool_key)
        return visible_tool_keys

    def filter_visible_tools(
            self,
            *,
            request_tools: list[Any],
            loaded_tool_keys: list[str] | None,
    ) -> list[Any]:
        """
        功能描述：
            过滤当前请求中的工具列表，仅保留当前可见的管理工具。

        参数说明：
            request_tools (list[Any]): 当前请求中的工具对象数组。
            loaded_tool_keys (list[str] | None): 当前状态中的已加载工具 key 数组。

        返回值：
            list[Any]:
                保留当前可见的管理工具，并始终保留非注册中心管理工具。

        异常说明：
            无。
        """

        visible_tool_keys = self.resolve_visible_tool_key_set(loaded_tool_keys)
        managed_tool_keys = self.get_managed_tool_key_set()
        visible_tools: list[Any] = []
        for tool_obj in request_tools:
            tool_key = str(getattr(tool_obj, "name", "") or "").strip()
            if not tool_key:
                visible_tools.append(tool_obj)
                continue
            if tool_key not in managed_tool_keys:
                visible_tools.append(tool_obj)
                continue
            if tool_key in visible_tool_keys:
                visible_tools.append(tool_obj)
        return visible_tools


CLIENT_COMMERCE_TOOL_REGISTRY = ClientCommerceToolRegistry()

__all__ = [
    "CLIENT_COMMERCE_TOOL_REGISTRY",
    "ClientCommerceToolRegistry",
]
