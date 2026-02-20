from __future__ import annotations

from typing import Any

from app.agent.admin.node.common import build_worker_update
from app.agent.admin.state import AgentState
from app.core.assistant_status import status_node
from app.core.langsmith import traceable


@status_node(
    node="excel",
    start_message="正在处理表格任务",
    display_when="always",
)
@traceable(name="Supervisor Excel Agent Node", run_type="chain")
def excel_agent(state: AgentState) -> dict[str, Any]:
    """
    Execute excel node (placeholder).
    """
    content = "表格能力暂未实现，请稍后重试。"
    return build_worker_update(
        state=state,
        node_name="excel_agent",
        result_key="excel",
        content=content,
        status="failed",
        model_name="unknown",
        input_messages=[],
        tool_calls=[],
        error="excel_agent 未实现",
    )

