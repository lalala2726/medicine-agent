from __future__ import annotations

from app.agent.admin.state import AgentState
from app.core.langsmith import traceable


_SUPERVISOR_PROMPT = """
    根据用户的描述完成用户的需求
"""


@traceable(name="Supervisor Agent Node", run_type="chain")
def supervisor_agent(state: AgentState) -> AgentState:
    pass
