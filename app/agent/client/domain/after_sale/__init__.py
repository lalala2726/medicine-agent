"""Client 售后域节点包。"""

from app.agent.client.domain.after_sale.node import after_sale_agent
from app.agent.client.domain.after_sale.tools import (
    check_after_sale_eligibility,
    get_after_sale_detail,
)
from app.agent.client.domain.tools.action_tools import (
    open_user_after_sale_list,
    open_user_order_list,
)

__all__ = [
    "after_sale_agent",
    "check_after_sale_eligibility",
    "get_after_sale_detail",
    "open_user_after_sale_list",
    "open_user_order_list",
]
