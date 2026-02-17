from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.admin.agent_utils import (
    NodeExecutionResult,
    build_standard_node_update,
    execute_text_node,
)
from app.agent.admin.agent_state import AgentState
from app.agent.admin.node.runtime_context import build_step_runtime
from app.core.assistant_status import status_node
from app.core.langsmith import traceable
from app.core.llm import create_chat_model
from app.schemas.prompt import base_prompt

_SUMMARY_PROMPT = (
        """
        你是药品商城后台的总结节点（summary_agent）。
        你的任务是把已有节点产出整理成对用户可直接阅读的最终结论。
        这边不要暴露节点的名称等信息
        听从上层节点的调度
        若无指令，简单整理内容即可，优先使用表格
    """
        + base_prompt
)


def _build_summary_input(state: AgentState, runtime: dict[str, Any]) -> dict[str, Any]:
    # summary 仅基于 DAG read_from 注入的上游输出做汇总，不再兼容旧直连上下文字段。
    payload: dict[str, Any] = {
        "task_description": runtime.get("task_description") or "汇总已有结果并生成最终结论",
        "upstream_outputs": runtime.get("upstream_outputs") or {},
        "read_from": runtime.get("read_from") or [],
        "errors": state.get("errors") or [],
    }
    user_input = runtime.get("user_input")
    if isinstance(user_input, str) and user_input:
        payload["user_input"] = user_input

    history_messages = runtime.get("history_messages_serialized")
    if isinstance(history_messages, list) and history_messages:
        payload["history_messages"] = history_messages

    return payload


@status_node(
    node="summary",
    start_message="正在汇总问题",
    display_when="after_coordinator",
)
@traceable(name="Summary Agent Node", run_type="chain")
def summary_agent(state: AgentState) -> dict:
    # runtime 包含当前步骤 task/read_from/final_output/context 开关。
    runtime = build_step_runtime(
        state,
        "summary_agent",
        default_task_description="汇总已有结果并生成最终结论",
    )
    summary_input = _build_summary_input(state, runtime)
    task_description = runtime.get("task_description") or "汇总已有结果并生成最终结论"
    final_output = bool(runtime.get("final_output"))
    has_data = bool(
        summary_input.get("upstream_outputs")
    )
    if not has_data and not summary_input["errors"]:
        execution_result = NodeExecutionResult(
            content="当前没有可汇总的业务结果。请先提供明确任务或先执行业务节点。",
            status="completed",
        )
        return build_standard_node_update(
            state=state,
            runtime=runtime,
            node_name="summary_agent",
            result_key="summary",
            execution_result=execution_result,
            is_end=final_output,
        )

    llm = create_chat_model(model="qwen-plus", temperature=0.2)
    messages = [
        SystemMessage(content=_SUMMARY_PROMPT),
        HumanMessage(
            content=(
                f"总结任务：{task_description}\n\n"
                f"输入数据：\n{json.dumps(summary_input, ensure_ascii=False)}"
            )
        ),
    ]
    execution_result = execute_text_node(
        llm=llm,
        messages=messages,
        fallback_content="总结节点暂时不可用，请稍后重试。",
        fallback_error="summary_agent 执行失败",
    )
    return build_standard_node_update(
        state=state,
        runtime=runtime,
        node_name="summary_agent",
        result_key="summary",
        execution_result=execution_result,
        is_end=final_output,
    )
