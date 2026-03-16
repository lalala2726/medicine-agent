from __future__ import annotations

from typing import Any, TypeAlias, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import MessagesState

ChatHistoryMessage: TypeAlias = HumanMessage | AIMessage | SystemMessage


class TokenCounterState(TypedDict):
    """统一 token 计数结构。"""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ToolCallTraceState(TypedDict, total=False):
    """单次工具调用追踪结构。"""

    tool_name: str
    tool_call_id: str | None
    tool_input: Any


class ExecutionTraceState(TypedDict, total=False):
    """单个节点执行追踪结构。"""

    sequence: int
    node_name: str
    model_name: str
    status: str
    output_text: str
    llm_usage_complete: bool
    llm_token_usage: TokenCounterState | None
    tool_calls: list[ToolCallTraceState]
    node_context: dict[str, Any] | None


class NodeTokenBreakdownState(TypedDict):
    """节点级 token 明细。"""

    node_name: str
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class TokenUsageState(TypedDict):
    """消息级 token 使用汇总。"""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    is_complete: bool
    node_breakdown: list[NodeTokenBreakdownState]


class GatewayRoutingState(TypedDict):
    """Gateway 路由结果结构。"""

    route_targets: list[str]
    task_difficulty: str


class AgentState(MessagesState, total=False):
    """Client agent 工作流状态。"""

    routing: GatewayRoutingState
    context: str
    history_messages: list[ChatHistoryMessage]
    execution_traces: list[ExecutionTraceState]
    token_usage: TokenUsageState | None
    result: str
