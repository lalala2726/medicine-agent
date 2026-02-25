"""
`app.agent.assistant.tools` 工具导出模块。

对外统一暴露后台管理助手可调用的工具函数与工具集合 `ADMIN_TOOLS`。
"""

from app.agent.assistant.tools.analytics_tool import (
    get_analytics_hot_products,
    get_analytics_order_status_distribution,
    get_analytics_order_trend,
    get_analytics_overview,
    get_analytics_payment_distribution,
    get_analytics_product_return_rates,
)
from app.agent.assistant.tools.base_tools import get_user_info
from app.agent.assistant.tools.chart_tool import get_chart_sample_by_name, get_supported_chart_types
from app.agent.assistant.tools.order_tool import get_order_list, get_orders_detail
from app.agent.assistant.tools.product_tool import get_drug_detail, get_product_detail, get_product_list

ADMIN_TOOLS = [
    get_user_info,
    get_product_list,
    get_product_detail,
    get_order_list,
    get_orders_detail,
    get_drug_detail,
    get_analytics_overview,
    get_analytics_order_trend,
    get_analytics_order_status_distribution,
    get_analytics_payment_distribution,
    get_analytics_hot_products,
    get_analytics_product_return_rates,
    get_supported_chart_types,
    get_chart_sample_by_name,
]

__all__ = [
    "ADMIN_TOOLS",
    "get_user_info",
    "get_product_list",
    "get_product_detail",
    "get_order_list",
    "get_orders_detail",
    "get_drug_detail",
    "get_analytics_overview",
    "get_analytics_order_trend",
    "get_analytics_order_status_distribution",
    "get_analytics_payment_distribution",
    "get_analytics_hot_products",
    "get_analytics_product_return_rates",
    "get_supported_chart_types",
    "get_chart_sample_by_name",
]
