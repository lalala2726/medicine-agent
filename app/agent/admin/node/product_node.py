from __future__ import annotations

from app.core.assistant_status import status_node
from app.core.langsmith import traceable
from app.schemas.prompt import base_prompt

_PRODUCT_SYSTEM_PROMPT = (
        """
        
        """
        + base_prompt
)


@status_node(
    node="product",
    start_message="正在处理商品问题",
    display_when="always",
)
@traceable(name="Supervisor Product Agent Node", run_type="chain")
def product_agent(tack_description: str) -> str:
    pass
