from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import MessagesState


ChatHistoryMessage = HumanMessage | AIMessage


def _merge_dict(
        left: dict[str, Any] | None,
        right: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Merge dict values for graph state reducers.
    """
    merged = dict(left or {})
    merged.update(right or {})
    return merged


def _merge_list(
        left: list[Any] | None,
        right: list[Any] | None,
) -> list[Any]:
    """
    Merge list values for graph state reducers.
    """
    return list(left or []) + list(right or [])


class AgentState(MessagesState, total=False):
    """
    Global state for gateway + supervisor workflow.
    """

    user_input: str
    next_node: str

    # Shared extracted context across nodes.
    context: Annotated[dict[str, Any], _merge_dict]

    # Routing metadata for graph decisions.
    routing: Annotated[dict[str, Any], _merge_dict]

    # Keep history for current service persistence flow (unchanged in this migration).
    history_messages: list[ChatHistoryMessage]

    results: Annotated[dict[str, Any], _merge_dict]
    execution_traces: Annotated[list[dict[str, Any]], _merge_list]
    errors: Annotated[list[str], _merge_list]

