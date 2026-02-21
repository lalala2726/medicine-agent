from __future__ import annotations

from typing import Any

from app.agent.admin.state import AgentState
from app.core.langsmith import traceable

_GATEWAY_PROMPT = """
    """


@traceable(name="Supervisor Gateway Router Node", run_type="chain")
def gateway_router(state: AgentState) -> dict[str, Any]:
    pass
