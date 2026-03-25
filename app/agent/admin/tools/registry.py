"""
管理端工具注册中心。
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from app.agent.admin.tools.after_sale import after_sale_detail, after_sale_list
from app.agent.admin.tools.analytics import (
    analytics_after_sale_efficiency_summary,
    analytics_after_sale_reason_distribution,
    analytics_after_sale_status_distribution,
    analytics_after_sale_trend,
    analytics_conversion_summary,
    analytics_fulfillment_summary,
    analytics_range_summary,
    analytics_realtime_overview,
    analytics_return_refund_risk_products,
    analytics_sales_trend,
    analytics_top_selling_products,
)
from app.agent.admin.tools.base import (
    create_request_tool_access_tool,
    get_safe_user_info,
    search_knowledge_context,
)
from app.agent.admin.tools.order import (
    order_detail,
    order_list,
    order_shipping,
    order_timeline,
)
from app.agent.admin.tools.product import drug_detail, product_detail, product_list
from app.agent.admin.tools.user import (
    user_consume_info,
    user_detail,
    user_list,
    user_wallet,
    user_wallet_flow,
)


class AdminToolRegistry:
    """
    功能描述：
        统一维护 admin 单 Agent 的基础工具、业务工具和工具索引。

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

        self._business_tools: tuple[BaseTool, ...] = (
            order_list,
            order_detail,
            order_timeline,
            order_shipping,
            product_list,
            product_detail,
            drug_detail,
            after_sale_list,
            after_sale_detail,
            user_list,
            user_detail,
            user_wallet,
            user_wallet_flow,
            user_consume_info,
            analytics_realtime_overview,
            analytics_range_summary,
            analytics_conversion_summary,
            analytics_fulfillment_summary,
            analytics_after_sale_efficiency_summary,
            analytics_after_sale_status_distribution,
            analytics_after_sale_reason_distribution,
            analytics_top_selling_products,
            analytics_return_refund_risk_products,
            analytics_sales_trend,
            analytics_after_sale_trend,
        )
        self._request_tool_access = create_request_tool_access_tool(
            get_allowed_tool_keys=self.get_business_tool_key_set,
        )
        self._base_tools: tuple[BaseTool, ...] = (
            self._request_tool_access,
            search_knowledge_context,
            get_safe_user_info,
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
            返回 create_agent 使用的全量已注册工具列表。

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

    def resolve_visible_tool_key_set(self, granted_tool_keys: list[str] | None) -> set[str]:
        """
        功能描述：
            计算当前模型可见的管理工具 key 集合。

        当前策略：
            admin 侧已允许全部工具直接使用，因此忽略 `granted_tool_keys`，
            始终返回全部受管理工具 key。

        参数说明：
            granted_tool_keys (list[str] | None): 当前状态中的已授权工具 key 数组。

        返回值：
            set[str]: 当前应暴露给模型的管理工具 key 集合。

        异常说明：
            无。
        """

        _ = granted_tool_keys
        return self.get_managed_tool_key_set()

    def filter_visible_tools(
            self,
            *,
            request_tools: list[Any],
            granted_tool_keys: list[str] | None,
    ) -> list[Any]:
        """
        功能描述：
            过滤当前请求中的工具列表，仅隐藏未授权的管理工具。

        参数说明：
            request_tools (list[Any]): 当前请求中的工具对象数组。
            granted_tool_keys (list[str] | None): 当前状态中的已授权工具 key 数组。

        返回值：
            list[Any]:
                当前策略下保留全部工具；
                包括管理工具与非注册中心管理工具（例如 skill 工具）。

        异常说明：
            无。
        """

        _ = granted_tool_keys
        return list(request_tools)


# 默认管理端工具注册中心。
ADMIN_TOOL_REGISTRY = AdminToolRegistry()

__all__ = [
    "ADMIN_TOOL_REGISTRY",
    "AdminToolRegistry",
]
