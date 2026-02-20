from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.admin.agent_utils import build_execution_trace_update, serialize_messages
from app.agent.admin.state import AgentState
from app.core.langsmith import traceable
from app.core.llm import create_chat_model

# Supervisor 允许下发的下一跳节点集合，FINISH 表示流程终止。
_ALLOWED_NEXT_NODES = {
    "order_agent",
    "product_agent",
    "excel_agent",
    "FINISH",
}

# 同一节点最大允许被 supervisor 调用次数，用于循环保护。
_MAX_NODE_CALLS = 2

_SUPERVISOR_PROMPT = """
    你是药品商城后台的动态主管 Supervisor。
    你每次只负责选择“下一步调用哪个节点”，不是一次性规划完整流程。
    
    你只能输出 JSON，格式必须是：
    {"next_node":"order_agent|product_agent|excel_agent|FINISH","directive":"给下游节点的可执行指令"}
    
    约束：
    1. 优先使用现有 context 中的结构化信息，避免重复向用户索要同一参数。
    2. 当 next_node 为 order_agent/product_agent/excel_agent 时，directive 必须是可执行指令且不能为空。
    3. 当任务已完成、无法继续推进、或继续执行价值不高时，输出 FINISH 且 directive 可为空字符串。
    4. 不要输出任何解释文本、Markdown、代码块。
"""


def _resolve_supervisor_decision(payload: dict[str, Any]) -> tuple[str, str]:
    """
    校验 supervisor 原始 JSON 输出并归一化为受控决策。

    Args:
        payload: 模型返回的 JSON 对象。

    Returns:
        tuple[str, str]: 归一化决策 `(next_node, directive)`。
            - 若 `next_node` 非法，回退为 `("FINISH", "")`；
            - 若 `next_node` 不是 `FINISH` 且 `directive` 为空，也回退为 `("FINISH", "")`；
            - `FINISH` 场景下 directive 始终归一为空字符串。
    """

    candidate = str(payload.get("next_node") or "").strip()
    if candidate not in _ALLOWED_NEXT_NODES:
        return "FINISH", ""

    directive = str(payload.get("directive") or "").strip()
    if candidate != "FINISH" and not directive:
        return "FINISH", ""
    if candidate == "FINISH":
        return "FINISH", ""
    return candidate, directive


@traceable(name="Supervisor Agent Node", run_type="chain")
def supervisor_agent(state: AgentState) -> dict[str, Any]:
    """
    执行动态主管决策：每轮仅选择下一跳节点并下发指令。

    Args:
        state: 当前图状态，需包含：
            - `messages`: 全量历史消息（本函数按你确认策略使用全量）；
            - `context`: 共享上下文（含节点调用计数、结构化提取信息）；
            - `routing`: 路由元信息（mode/turn 等）。

    Returns:
        dict[str, Any]: supervisor 的状态增量更新，字段说明如下：
            - `next_node`: 下一跳节点名或 `FINISH`；
            - `routing`: 更新后的路由元信息（含 `turn/finished/directive/route_target`）；
            - `context.node_call_counts`: 叠加后的节点调用次数；
            - `execution_traces`: 当前轮决策追踪记录。
    """

    context = dict(state.get("context") or {})
    routing = dict(state.get("routing") or {})
    counts = dict(context.get("node_call_counts") or {})
    all_messages = list(state.get("messages") or [])

    supervisor_input = {
        "user_input": state.get("user_input"),
        "context": context,
        "routing": routing,
        "messages": serialize_messages(all_messages),
    }

    model_name = "qwen-flash"
    input_messages: list[Any] = [
        SystemMessage(content=_SUPERVISOR_PROMPT),
        HumanMessage(content=json.dumps(supervisor_input, ensure_ascii=False, default=str)),
    ]

    next_node = "FINISH"
    directive = ""
    try:
        llm = create_chat_model(
            model=model_name,
            temperature=0,
            response_format={"type": "json_object"},
        )
        response = llm.invoke(input_messages)
        payload = json.loads(str(response.content))
        next_node, directive = _resolve_supervisor_decision(payload)
    except Exception:
        next_node, directive = "FINISH", ""

    # 循环保护：同一节点调用达到阈值后直接 FINISH。
    if next_node != "FINISH" and int(counts.get(next_node, 0)) >= _MAX_NODE_CALLS:
        next_node = "FINISH"
        directive = ""

    if next_node != "FINISH":
        counts[next_node] = int(counts.get(next_node, 0)) + 1

    context["node_call_counts"] = counts
    routing["turn"] = int(routing.get("turn") or 0) + 1
    routing["next_node"] = next_node
    routing["finished"] = next_node == "FINISH"
    routing["directive"] = directive
    if next_node != "FINISH":
        routing["route_target"] = next_node
    else:
        routing["route_target"] = "supervisor_agent"

    update: dict[str, Any] = {
        "routing": routing,
        "context": context,
        "next_node": next_node,
    }
    update.update(
        build_execution_trace_update(
            node_name="supervisor_agent",
            model_name=model_name,
            input_messages=serialize_messages(input_messages),
            output_text=json.dumps(update, ensure_ascii=False, default=str),
            tool_calls=[],
        )
    )
    return update
