from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.admin.agent_utils import build_execution_trace_update, serialize_messages
from app.agent.admin.model_policy import (
    DEFAULT_NODE_GOAL,
    normalize_task_difficulty,
    resolve_model_profile,
)
from app.agent.admin.state import AgentState
from app.core.assistant_status import emit_thinking_notice
from app.core.langsmith import traceable
from app.core.llm import create_chat_model
from app.schemas.prompt import base_prompt
from app.utils.streaming_utils import invoke, stream_with_reasoning

_SUMMARY_SYSTEM_PROMPT = (
    """
    你是药品商城后台管理助手的最终汇总节点（summary_agent）。
    当前所有可执行业务步骤已经结束，你只负责根据输入数据给用户最终回复。
    
    硬约束：
    1. 必须围绕 `node_goal` 的重点组织输出，不能只贴原始数据。
    2. 不要复述内部调度流程，不要提到 supervisor/target_node/node_goal 等内部字段名。
    3. 若现有数据不足以完成用户诉求，明确告知“当前无法做到”的具体原因，并给出可执行下一步。
    4. 不要编造未出现的数据；未知信息请显式说明“未获取到”。
    
    你输出的是直接给用户的最终文本，优先markdown和表格方式
"""
    + base_prompt
)


def _build_summary_messages(state: AgentState) -> list[Any]:
    """
    构建 summary 节点的模型输入消息。

    Args:
        state: 当前图状态，至少包含 `routing.node_goal`、`context`、`results`、
            `errors` 与 `user_input` 字段。

    Returns:
        list[Any]: 两条消息组成的输入列表：
            - `SystemMessage`：汇总规则提示词；
            - `HumanMessage`：结构化汇总上下文 JSON（含用户诉求、汇总指令、结果与错误）。
    """

    routing = dict(state.get("routing") or {})
    context = dict(state.get("context") or {})
    node_goal = str(routing.get("node_goal") or "").strip() or DEFAULT_NODE_GOAL
    payload = {
        "user_input": state.get("user_input"),
        "node_goal": node_goal,
        "context": {
            "agent_outputs": context.get("agent_outputs") or {},
            "last_agent": context.get("last_agent"),
            "last_agent_response": context.get("last_agent_response"),
            "shared_context": {
                key: value
                for key, value in context.items()
                if key not in {"agent_outputs", "last_agent", "last_agent_response"}
            },
        },
        "results": state.get("results") or {},
        "errors": state.get("errors") or [],
        "messages_tail": serialize_messages(list(state.get("messages") or [])[-8:]),
    }
    return [
        SystemMessage(content=_SUMMARY_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False, default=str)),
    ]


@traceable(name="Supervisor Summary Agent Node", run_type="chain")
def summary_agent(state: AgentState) -> dict[str, Any]:
    """
    执行最终汇总节点，输出面向用户的最终结论文本。

    Args:
        state: 当前图状态，要求包含 Supervisor 写入的汇总目标（`routing.node_goal`）
            以及各 worker 的聚合产出（`context.agent_outputs` / `results` / `errors`）。

    Returns:
        dict[str, Any]: Summary 节点状态增量，核心字段如下：
            - `results.summary`: 汇总输出内容（`mode=summary`，`is_end=True`）；
            - `messages`: 新增一条 AIMessage（最终回复）；
            - `context.last_agent/last_agent_response`: 最近输出节点信息；
            - `execution_traces`: 本节点输入输出追踪信息。
    """

    routing = dict(state.get("routing") or {})
    task_difficulty = normalize_task_difficulty(routing.get("task_difficulty"))
    model_profile = resolve_model_profile(task_difficulty)
    messages = _build_summary_messages(state)

    model_name = str(model_profile.get("model") or "qwen-plus")
    think_enabled = bool(model_profile.get("think"))
    content = "当前需求暂时无法完成。请补充更明确的条件后重试。"

    try:
        llm = create_chat_model(
            model=model_name,
            think=think_enabled,
            temperature=0.3,
        )
        if hasattr(llm, "stream") and callable(getattr(llm, "stream")):
            answer_chunks, reasoning_chunks = stream_with_reasoning(llm, messages)
            if think_enabled and reasoning_chunks:
                emit_thinking_notice(
                    node="summary_agent",
                    state="thinking_start",
                    meta={
                        "model": model_name,
                        "task_difficulty": task_difficulty,
                    },
                )
                for chunk in reasoning_chunks:
                    emit_thinking_notice(
                        node="summary_agent",
                        state="thinking_delta",
                        text=chunk,
                        meta={
                            "model": model_name,
                            "task_difficulty": task_difficulty,
                        },
                    )
                emit_thinking_notice(
                    node="summary_agent",
                    state="thinking_end",
                    meta={
                        "model": model_name,
                        "task_difficulty": task_difficulty,
                    },
                )
            content = "".join(answer_chunks).strip() or invoke(llm, messages)
        else:
            content = invoke(llm, messages)
    except Exception:
        pass

    results = dict(state.get("results") or {})
    results["summary"] = {
        "mode": "summary",
        "content": content,
        "is_end": True,
    }
    update: dict[str, Any] = {
        "results": results,
        "messages": [AIMessage(content=content)],
        "context": {
            "last_agent": "summary_agent",
            "last_agent_response": content,
        },
    }
    update.update(
        build_execution_trace_update(
            node_name="summary_agent",
            model_name=model_name,
            input_messages=serialize_messages(messages),
            output_text=content,
            tool_calls=[],
        )
    )
    return update
