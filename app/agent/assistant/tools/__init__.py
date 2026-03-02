"""
`app.agent.assistant.tools` 工具导出模块。

对外统一暴露后台管理助手可调用的工具函数与工具集合 `ADMIN_TOOLS`。
"""

from app.agent.assistant.tools.after_sale_tools import (
    get_admin_after_sale_detail,
    get_admin_after_sale_list,
)
from app.agent.assistant.tools.analytics_tools import (
    get_analytics_hot_products,
    get_analytics_order_status_distribution,
    get_analytics_order_trend,
    get_analytics_overview,
    get_analytics_payment_distribution,
    get_analytics_product_return_rates,
)
from app.agent.assistant.tools.base_tools import get_safe_user_info
from app.agent.assistant.tools.order_tools import (
    get_order_list,
    get_order_shipping,
    get_order_timeline,
    get_orders_detail,
)
from app.agent.assistant.tools.product_tools import get_drug_detail, get_product_detail, get_product_list
from app.agent.assistant.tools.user_tools import (
    get_admin_user_consume_info,
    get_admin_user_detail,
    get_admin_user_list,
    get_admin_user_wallet,
    get_admin_user_wallet_flow,
)

ADMIN_TOOLS = [
    get_safe_user_info,
    get_product_list,
    get_product_detail,
    get_order_list,
    get_orders_detail,
    get_order_timeline,
    get_order_shipping,
    get_drug_detail,
    get_analytics_overview,
    get_analytics_order_trend,
    get_analytics_order_status_distribution,
    get_analytics_payment_distribution,
    get_analytics_hot_products,
    get_analytics_product_return_rates,
    get_admin_after_sale_list,
    get_admin_after_sale_detail,
    get_admin_user_list,
    get_admin_user_detail,
    get_admin_user_wallet,
    get_admin_user_wallet_flow,
    get_admin_user_consume_info,
]

__all__ = [
    "ADMIN_TOOLS",
    "get_safe_user_info",
    "get_product_list",
    "get_product_detail",
    "get_order_list",
    "get_orders_detail",
    "get_order_timeline",
    "get_order_shipping",
    "get_drug_detail",
    "get_analytics_overview",
    "get_analytics_order_trend",
    "get_analytics_order_status_distribution",
    "get_analytics_payment_distribution",
    "get_analytics_hot_products",
    "get_analytics_product_return_rates",
    "get_admin_after_sale_list",
    "get_admin_after_sale_detail",
    "get_admin_user_list",
    "get_admin_user_detail",
    "get_admin_user_wallet",
    "get_admin_user_wallet_flow",
    "get_admin_user_consume_info",
]
