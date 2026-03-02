"""
`app.agent.assistant.sub_agents` 子代理导出模块。

统一导出包含模型调用能力的领域子代理入口，供 supervisor 节点调度。
"""

from app.agent.assistant.sub_agents.after_sale_sub_agent import after_sale_sub_agent
from app.agent.assistant.sub_agents.analytics_sub_agent import analytics_sub_agent
from app.agent.assistant.sub_agents.order_sub_agent import order_sub_agent
from app.agent.assistant.sub_agents.product_sub_agent import product_sub_agent
from app.agent.assistant.sub_agents.user_sub_agent import user_sub_agent

__all__ = [
    "order_sub_agent",
    "after_sale_sub_agent",
    "product_sub_agent",
    "analytics_sub_agent",
    "user_sub_agent",
]
