"""Client 通用包兼容导出。"""

from app.agent.client.domain.chat import chat_agent
from app.agent.client.domain.router import gateway_router
from app.agent.client.domain.tools import (
    open_user_after_sale_list,
    open_user_order_list,
    send_product_card,
)

__all__ = [
    "chat_agent",
    "gateway_router",
    "open_user_after_sale_list",
    "open_user_order_list",
    "send_product_card",
]
