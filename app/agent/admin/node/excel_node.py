from app.agent.admin.agent_state import AgentState
from app.agent.admin.node.runtime_context import build_step_output_update, build_step_runtime


def excel_agent(state: AgentState) -> dict:
    runtime = build_step_runtime(
        state,
        "excel_agent",
        default_task_description="处理表格相关任务",
    )
    content = "表格能力暂未实现，请稍后重试。"
    result: dict = {}
    result.update(
        build_step_output_update(
            runtime,
            node_name="excel_agent",
            status="failed",
            text=content,
            output={},
            error="excel_agent 未实现",
        )
    )
    return result
