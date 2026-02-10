from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.admin.agent_state import AgentState
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


def _build_summary_input(state: AgentState) -> dict[str, Any]:
    return {
        "order_context": state.get("order_context") or {},
        "excel_context": state.get("excel_context") or {},
        "aftersale_context": state.get("aftersale_context") or {},
        "results": state.get("results") or {},
        "errors": state.get("errors") or [],
    }


@traceable(name="Summary Agent Node", run_type="chain")
def summary_agent(state: AgentState) -> dict:
    routing = state.get("routing") or {}
    current_step_map = routing.get("current_step_map") or {}
    step = (
        current_step_map.get("summary_agent")
        if isinstance(current_step_map, dict)
        else {}
    )
    task_description = (step or {}).get(
        "task_description"
    ) or "汇总已有结果并生成最终结论"

    summary_input = _build_summary_input(state)
    has_data = bool(
        summary_input["order_context"]
        or summary_input["excel_context"]
        or summary_input["results"]
    )
    if not has_data and not summary_input["errors"]:
        content = "当前没有可汇总的业务结果。请先提供明确任务或先执行业务节点。"
        results = dict(state.get("results") or {})
        results["summary"] = {"content": content}
        return {"results": results}

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
    except Exception:
        content = "总结节点暂时不可用，请稍后重试。"

    results = dict(state.get("results") or {})
    results["summary"] = {"content": content}
    return {"results": results}
