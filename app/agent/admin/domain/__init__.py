"""管理助手 domain 分包导出模块。"""

from app.agent.admin.domain.after_sale.node import after_sale_agent
from app.agent.admin.domain.analytics.node import analytics_agent
from app.agent.admin.domain.common.adaptive_node import adaptive_agent
from app.agent.admin.domain.common.chat_node import chat_agent
from app.agent.admin.domain.common.gateway_node import gateway_router
from app.agent.admin.domain.order.node import order_agent
from app.agent.admin.domain.product.node import product_agent
from app.agent.admin.domain.user.node import user_agent

__all__ = [
    "gateway_router",
    "chat_agent",
    "order_agent",
    "product_agent",
    "after_sale_agent",
    "user_agent",
    "analytics_agent",
    "adaptive_agent",
]
