from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from app.agent.admin.state import AgentState
from app.core.langsmith import traceable

_SUPERVISOR_PROMPT = """
    根据用户的描述完成用户的需求
"""


@traceable(name="Supervisor Agent Node", run_type="chain")
def supervisor_agent(state: AgentState) -> dict[str, Any]:
    user_input = str(state.get("user_input") or "").strip()
    content = f"已进入业务处理节点（supervisor）。用户问题：{user_input}" if user_input else "已进入业务处理节点（supervisor）。"

    return {
        "result": content,
        "messages": [AIMessage(content=content)],
    }
