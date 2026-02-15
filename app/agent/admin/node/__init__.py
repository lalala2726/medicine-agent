"""
药品商城后台管理助手节点包。

该包包含管理工作流程中使用的所有智能体节点：
- chat_agent: 处理一般聊天和非业务查询
- chart_agent: 从结构化数据生成图表
- coordinator: 协调多节点工作流程和任务规划
- excel_agent: 处理Excel文件处理和数据提取
- order_agent: 处理订单相关的业务流程
- summary_agent: 汇总多个节点的结果

每个节点都使用 @status_node 装饰器来跟踪执行状态，并使用 @traceable 装饰器来支持 LangSmith 追踪。
"""

from app.agent.admin.node.chat_node import chat_agent
from app.agent.admin.node.chart_node import chart_agent
from app.agent.admin.node.coordinator_node import coordinator
from app.agent.admin.node.excel_node import excel_agent
from app.agent.admin.node.order_node import order_agent
from app.agent.admin.node.product_node import product_agent
from app.agent.admin.node.summary_node import summary_agent

__all__ = [
    "chat_agent",
    "chart_agent",
    "coordinator",
    "excel_agent",
    "order_agent",
    "product_agent",
    "summary_agent",
]
