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
    执行表格节点占位逻辑。

    当前版本用于保留 supervisor 可调度链路，统一返回“暂未实现”结果，
    以便上层能在慢车道中感知该步骤状态并继续决策或结束。

    Args:
        state: 当前图状态。

    Returns:
        dict[str, Any]: 标准 worker 更新结构，包含：
            - `results.excel`：失败提示内容；
            - `context`：最近节点输出信息；
            - `messages`：一条 AI 提示消息；
            - `execution_traces`：占位执行追踪；
            - `errors`：固定写入“未实现”错误信息。
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
