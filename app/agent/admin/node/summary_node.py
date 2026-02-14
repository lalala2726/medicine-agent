from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.admin.agent_state import AgentState
from app.agent.admin.node.runtime_context import (
    build_step_output_update,
    build_step_runtime,
)
from app.core.assistant_status import status_node
from app.core.langsmith import traceable
from app.core.llm import create_chat_model
from app.schemas.prompt import base_prompt
from app.utils.streaming_utils import invoke

_SUMMARY_PROMPT = (
        """
        你是药品商城后台的总结节点（summary_agent）。
        你的任务是把已有节点产出整理成对用户可直接阅读的最终结论。
        
        输出要求：
        1. 先给结论，再给关键依据。
        2. 若存在不确定项，明确标注“待确认”。
        3. 保持简洁、准确，不编造不存在的数据。
    """
        + base_prompt
)


def _build_summary_input(state: AgentState, runtime: dict[str, Any]) -> dict[str, Any]:
    # summary 在 coordinator 模式下优先基于 read_from 读取上游步骤输出，
    # 只在直连模式下兼容读取旧字段（order_context/results 等）。
    payload: dict[str, Any] = {
        "task_description": runtime.get("task_description") or "汇总已有结果并生成最终结论",
        "upstream_outputs": runtime.get("upstream_outputs") or {},
        "read_from": runtime.get("read_from") or [],
        "errors": state.get("errors") or [],
    }
    user_input = runtime.get("user_input")
    if isinstance(user_input, str) and user_input:
        payload["user_input"] = user_input

    history_messages = runtime.get("history_messages")
    if isinstance(history_messages, list) and history_messages:
        payload["history_messages"] = history_messages

    if not runtime.get("coordinator_mode"):
        payload["order_context"] = state.get("order_context") or {}
        payload["excel_context"] = state.get("excel_context") or {}
        payload["aftersale_context"] = state.get("aftersale_context") or {}
        payload["results"] = state.get("results") or {}
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
    has_data = bool(
        summary_input.get("upstream_outputs")
        or summary_input.get("order_context")
        or summary_input.get("excel_context")
        or summary_input.get("results")
    )
    if not has_data and not summary_input["errors"]:
        # 早返回也要写 step_outputs，避免 planner 误判该步骤未执行。
        content = "当前没有可汇总的业务结果。请先提供明确任务或先执行业务节点。"
        results = dict(state.get("results") or {})
        results["summary"] = {"content": content}
        result = {"results": results}
        result.update(
            build_step_output_update(
                runtime,
                node_name="summary_agent",
                status="completed",
                text=content,
                output={"summary": results["summary"]},
            )
        )
        return result

    failed_error: str | None = None
    try:
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
        content = invoke(llm, messages)
        step_status = "completed"
    except Exception:
        content = "总结节点暂时不可用，请稍后重试。"
        step_status = "failed"
        failed_error = "summary_agent 执行失败"

    results = dict(state.get("results") or {})
    results["summary"] = {"content": content}
    # 标准化写回 step_outputs，供后续调度或最终兜底输出使用。
    result: dict[str, Any] = {"results": results}
    result.update(
        build_step_output_update(
            runtime,
            node_name="summary_agent",
            status=step_status,
            text=content,
            output={"summary": results["summary"]},
            error=failed_error,
        )
    )
    return result
