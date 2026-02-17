from app.agent.admin.agent_utils import NodeExecutionResult, build_standard_node_update
from app.agent.admin.agent_state import AgentState
from app.agent.admin.node.runtime_context import build_step_runtime


def excel_agent(state: AgentState) -> dict:
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
