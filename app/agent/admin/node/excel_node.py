from app.agent.admin.agent_utils import NodeExecutionResult, build_standard_node_update
from app.agent.admin.agent_state import AgentState
from app.agent.admin.node.runtime_context import build_step_runtime


def excel_agent(state: AgentState) -> dict:
    """
    执行 excel 占位节点。

    当前节点尚未实现真实业务能力，会固定返回失败态，避免流程静默成功。

    Args:
        state: 当前全局状态。

    Returns:
        dict: 标准节点增量更新（results、step_outputs、execution_traces）。
    """
    runtime = build_step_runtime(
        state,
        "excel_agent",
        default_task_description="处理表格相关任务",
    )
    execution_result = NodeExecutionResult(
        content="表格能力暂未实现，请稍后重试。",
        status="failed",
        error="excel_agent 未实现",
    )
    return build_standard_node_update(
        state=state,
        runtime=runtime,
        node_name="excel_agent",
        result_key="excel",
        execution_result=execution_result,
        is_end=bool(runtime.get("final_output")),
    )
