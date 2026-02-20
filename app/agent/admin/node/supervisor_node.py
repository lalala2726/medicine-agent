from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.admin.agent_utils import build_execution_trace_update
from app.agent.admin.node.common import serialize_messages
from app.agent.admin.state import AgentState
from app.core.langsmith import traceable
from app.core.llm import create_chat_model

_ALLOWED_NEXT_NODES = {
    "order_agent",
    "product_agent",
    "excel_agent",
    "FINISH",
}
_MAX_NODE_CALLS = 2

_SUPERVISOR_PROMPT = """
你是药品商城后台的动态主管 Supervisor。
你每次只负责选择“下一步调用哪个节点”，不是一次性规划完整流程。

你只能输出 JSON，格式必须是：
{"next_node":"order_agent|product_agent|excel_agent|FINISH"}

约束：
1. 优先使用现有 context 中的结构化信息，避免重复向用户索要同一参数。
2. 当任务已完成、无法继续推进、或继续执行价值不高时，输出 FINISH。
3. 不要输出任何解释文本、Markdown、代码块。
"""


def _safe_next_node(payload: dict[str, Any]) -> str:
    candidate = str(payload.get("next_node") or "").strip()
    if candidate in _ALLOWED_NEXT_NODES:
        return candidate
    return "FINISH"


def _serialize_for_supervisor(messages: list[Any]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for message in messages:
        serialized.append(
            {
                "role": str(getattr(message, "type", "") or message.__class__.__name__).strip().lower() or "unknown",
                "content": getattr(message, "content", ""),
            }
        )
    return serialized


@traceable(name="Supervisor Agent Node", run_type="chain")
def supervisor_agent(state: AgentState) -> dict[str, Any]:
    """
    Dynamic supervisor: decide only the next node.
    """
    context = dict(state.get("context") or {})
    routing = dict(state.get("routing") or {})
    counts = dict(context.get("node_call_counts") or {})
    tail_messages = list(state.get("messages") or [])[-8:]

    supervisor_input = {
        "user_input": state.get("user_input"),
        "context": context,
        "routing": routing,
        "messages_tail": _serialize_for_supervisor(tail_messages),
    }

    model_name = "qwen-flash"
    input_messages: list[Any] = [
        SystemMessage(content=_SUPERVISOR_PROMPT),
        HumanMessage(content=json.dumps(supervisor_input, ensure_ascii=False, default=str)),
    ]

    next_node = "FINISH"
    try:
        llm = create_chat_model(
            model=model_name,
            temperature=0,
            response_format={"type": "json_object"},
        )
        response = llm.invoke(input_messages)
        payload = json.loads(str(response.content))
        next_node = _safe_next_node(payload)
    except Exception:
        next_node = "FINISH"

    # Loop guard: same node can be called at most 2 times.
    if next_node != "FINISH" and int(counts.get(next_node, 0)) >= _MAX_NODE_CALLS:
        next_node = "FINISH"

    if next_node != "FINISH":
        counts[next_node] = int(counts.get(next_node, 0)) + 1

    context["node_call_counts"] = counts
    routing["turn"] = int(routing.get("turn") or 0) + 1
    routing["next_node"] = next_node
    routing["finished"] = next_node == "FINISH"
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

