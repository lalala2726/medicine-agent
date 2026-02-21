"""
Supervisor 工作流节点导出模块。

统一导出 gateway、supervisor 与可复用子工具，供 `workflow.py` 与节点协作时集中引用。
"""

from app.agent.assistant.node.chat_node import chat_agent
from app.agent.assistant.node.gateway_node import gateway_router
from app.agent.assistant.node.supervisor_node import supervisor_agent
from app.agent.assistant.tools.analytics_tool import analytics_agent, analytics_tool_agent
from app.agent.assistant.tools.chart_tool import chart_agent, chart_tool_agent
from app.agent.assistant.tools.order_tool import order_agent, order_tool_agent
from app.agent.assistant.tools.product_tool import product_agent, product_tool_agent

__all__ = [
    "gateway_router",
    "supervisor_agent",
    "order_tool_agent",
    "product_tool_agent",
    "analytics_tool_agent",
    "chart_tool_agent",
    # 兼容旧导出名
    "order_agent",
    "product_agent",
    "analytics_agent",
    "chart_agent",
    "chat_agent",
]
