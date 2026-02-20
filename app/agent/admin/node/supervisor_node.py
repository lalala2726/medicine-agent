from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.admin.agent_utils import build_execution_trace_update, serialize_messages
from app.agent.admin.model_policy import (
    NORMAL_DIFFICULTY,
    apply_model_profile_to_routing,
    build_supervisor_decision,
    normalize_task_difficulty,
    resolve_model_profile,
)
from app.agent.admin.state import AgentState
from app.core.assistant_status import emit_sse_response, has_status_emitter
from app.core.langsmith import traceable
from app.core.llm import create_chat_model
from app.schemas.sse_response import AssistantResponse, Content, MessageType

# 同一节点最大允许被 supervisor 调用次数，用于循环保护。
_MAX_NODE_CALLS = 2

_SUPERVISOR_PROMPT = """
你是药品商城后台的动态主管 Supervisor。
你每次只负责选择“下一步调用哪个节点”，不是一次性规划完整流程。

你只能输出 JSON，格式必须是：
{"next_node":"order_agent|product_agent|excel_agent|FINISH","directive":"给下游节点的可执行指令","task_difficulty":"simple|normal|complex"}

约束：
1. 优先使用现有 context 中的结构化信息，避免重复向用户索要同一参数。
2. 当 next_node 为 order_agent/product_agent/excel_agent 时，directive 必须是可执行指令且不能为空。
3. 当任务已完成、无法继续推进、或继续执行价值不高时，输出 FINISH 且 directive 可为空字符串。
4. task_difficulty 必须输出 simple/normal/complex 之一；如不确定输出“normal”。
5. 不要输出任何解释文本、Markdown、代码块。
"""


def _emit_supervisor_enter_notice(routing: dict[str, Any]) -> None:
    """
    发射“进入 Supervisor 节点”的通知事件。

    Args:
        routing: 当前 routing 元信息，用于补充 turn/难度等上下文。

    Returns:
        None: 仅用于发射 SSE NOTICE 事件。
    """

    if not has_status_emitter():
        return

    emit_sse_response(
        AssistantResponse(
            content=Content(
                node="supervisor_agent",
                state="start",
                message="进入 Supervisor 动态决策",
            ),
            type=MessageType.NOTICE,
            meta={
                "turn": int(routing.get("turn") or 0),
                "task_difficulty": routing.get("task_difficulty"),
            },
        )
    )


def _emit_supervisor_dispatch_notice(
        *,
        next_node: str,
        directive: str,
        routing: dict[str, Any],
) -> None:
    """
    发射 Supervisor 决策结果通知，向前端明确下一跳节点信息。

    Args:
        next_node: Supervisor 决策出的下一跳节点（或 `FINISH`）。
        directive: 下发给 worker 的执行指令文本。
        routing: 已更新的 routing 元信息（含模型与难度）。

    Returns:
        None: 仅用于发射 SSE NOTICE 事件。
    """

    if not has_status_emitter():
        return

    message = (
        "Supervisor 判断任务完成，准备结束流程"
        if next_node == "FINISH"
        else f"Supervisor 下一跳节点: {next_node}"
    )
    emit_sse_response(
        AssistantResponse(
            content=Content(
                node="supervisor_agent",
                state="dispatch",
                message=message,
                name=next_node,
                arguments=directive or None,
            ),
            type=MessageType.NOTICE,
            meta={
                "next_node": next_node,
                "directive": directive,
                "task_difficulty": routing.get("task_difficulty"),
                "selected_model": routing.get("selected_model"),
                "think_enabled": routing.get("think_enabled"),
                "turn": routing.get("turn"),
            },
        )
    )


@traceable(name="Supervisor Agent Node", run_type="chain")
def supervisor_agent(state: AgentState) -> dict[str, Any]:
    """
    执行动态主管决策：每轮仅选择下一跳节点并下发指令。

    Args:
        state: 当前图状态，需包含：
            - `messages`: 全量历史消息；
            - `context`: 共享上下文（含节点调用计数、结构化提取信息）；
            - `routing`: 路由元信息（mode/turn/task_difficulty 等）。

    Returns:
        dict[str, Any]: supervisor 的状态增量更新，字段说明如下：
            - `next_node`: 下一跳节点名或 `FINISH`；
            - `routing`: 更新后的路由元信息（含 `turn/finished/directive/route_target/task_difficulty`）；
            - `context.node_call_counts`: 叠加后的节点调用次数；
            - `execution_traces`: 当前轮决策追踪记录。
    """

    context = dict(state.get("context") or {})
    routing = dict(state.get("routing") or {})
    counts = dict(context.get("node_call_counts") or {})
    all_messages = list(state.get("messages") or [])

    current_difficulty = normalize_task_difficulty(
        routing.get("task_difficulty") or NORMAL_DIFFICULTY
    )
    _emit_supervisor_enter_notice(routing)
    current_profile = resolve_model_profile(current_difficulty)

    supervisor_input = {
        "user_input": state.get("user_input"),
        "context": context,
        "routing": routing,
        "messages": serialize_messages(all_messages),
    }

    input_messages: list[Any] = [
        SystemMessage(content=_SUPERVISOR_PROMPT),
        HumanMessage(content=json.dumps(supervisor_input, ensure_ascii=False, default=str)),
    ]

    next_node = "FINISH"
    directive = ""
    task_difficulty = current_difficulty

    supervisor_model_name = str(current_profile.get("model") or "qwen-plus")
    supervisor_think_enabled = bool(current_profile.get("think"))
    try:
        llm = create_chat_model(
            model=supervisor_model_name,
            think=supervisor_think_enabled,
            temperature=0,
            response_format={"type": "json_object"},
        )
        response = llm.invoke(input_messages)
        payload = json.loads(str(response.content))
        next_node, directive, task_difficulty = build_supervisor_decision(
            payload,
            fallback_task_difficulty=current_difficulty,
        )
    except Exception:
        next_node, directive, task_difficulty = "FINISH", "", current_difficulty

    # 循环保护：同一节点调用达到阈值后直接 FINISH。
    if next_node != "FINISH" and int(counts.get(next_node, 0)) >= _MAX_NODE_CALLS:
        next_node = "FINISH"
        directive = ""

    if next_node != "FINISH":
        counts[next_node] = int(counts.get(next_node, 0)) + 1

    context["node_call_counts"] = counts

    resolved_profile = resolve_model_profile(task_difficulty)
    routing = apply_model_profile_to_routing(
        routing,
        task_difficulty=task_difficulty,
        profile=resolved_profile,
    )
    routing["turn"] = int(routing.get("turn") or 0) + 1
    routing["next_node"] = next_node
    routing["finished"] = next_node == "FINISH"
    routing["directive"] = directive
    if next_node != "FINISH":
        routing["route_target"] = next_node
    else:
        routing["route_target"] = "supervisor_agent"

    _emit_supervisor_dispatch_notice(
        next_node=next_node,
        directive=directive,
        routing=routing,
    )

    update: dict[str, Any] = {
        "routing": routing,
        "context": context,
        "next_node": next_node,
    }
    update.update(
        build_execution_trace_update(
            node_name="supervisor_agent",
            model_name=supervisor_model_name,
            input_messages=serialize_messages(input_messages),
            output_text=json.dumps(update, ensure_ascii=False, default=str),
            tool_calls=[],
        )
    )
    return update
