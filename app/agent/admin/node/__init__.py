"""
Supervisor workflow nodes.
"""

from app.agent.admin.node.chat_node import chat_agent
from app.agent.admin.node.excel_node import excel_agent
from app.agent.admin.node.gateway_node import gateway_router
from app.agent.admin.node.order_node import order_agent
from app.agent.admin.node.product_node import product_agent
from app.agent.admin.node.supervisor_node import supervisor_agent

__all__ = [
    "gateway_router",
    "supervisor_agent",
    "order_agent",
    "product_agent",
    "excel_agent",
    "chat_agent",
]

