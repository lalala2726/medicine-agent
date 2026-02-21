from __future__ import annotations

from app.core.assistant_status import status_node
from app.core.langsmith import traceable
from app.schemas.prompt import base_prompt

# 订单节点系统提示词：约束该节点只处理订单域任务，并按执行模式使用历史或主管指令。
_ORDER_SYSTEM_PROMPT = (
        """
    
        """
        + base_prompt
)


@status_node(
    node="order",
    start_message="正在处理订单问题",
    display_when="always",
)
@traceable(name="Supervisor Order Agent Node", run_type="chain")
def order_agent(tack_description: str) -> str:
    pass
