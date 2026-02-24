from __future__ import annotations

from typing import Any, Literal, TypeAlias, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import MessagesState

ChatHistoryMessage: TypeAlias = HumanMessage | AIMessage


class TokenCounterState(TypedDict):
    """统一 token 计数结构。"""

    # 输入 token 数。
    prompt_tokens: int
    # 输出 token 数。
    completion_tokens: int
    # 总 token 数。
    total_tokens: int


class ToolCallTraceState(TypedDict, total=False):
    """单次工具调用追踪结构（用于 state.execution_traces）。"""

    # 工具名称。
    tool_name: str
    # 工具输入参数（仅保留输入，不保留输出）。
    tool_input: Any
    # 是否调用失败。
    is_error: bool
    # 失败信息。
    error_message: str | None
    # 工具内部是否触发了 LLM。
    llm_used: bool
    # 工具内部 LLM usage 是否完整。
    llm_usage_complete: bool
    # 工具内部 LLM token 使用量。
    llm_token_usage: TokenCounterState | None
    # 子工具调用轨迹。
    children: list["ToolCallTraceState"]


class ExecutionTraceState(TypedDict, total=False):
    """单个节点执行追踪结构。"""

    # 节点名称（如 gateway_router/chat_agent/supervisor_agent）。
    node_name: str
    # 节点调用的模型名称。
    model_name: str
    # 节点输入消息（序列化后结构）。
    input_messages: list[Any]
    # 节点输出文本。
    output_text: str
    # 节点是否触发了 LLM。
    llm_used: bool
    # 节点 LLM usage 是否完整。
    llm_usage_complete: bool
    # 节点自身 LLM token 使用量。
    llm_token_usage: TokenCounterState | None
    # 节点下发生的工具调用轨迹。
    tool_calls: list[ToolCallTraceState]


class NodeTokenBreakdownState(TypedDict):
    """节点级 token 明细。"""

    # 节点名称。
    node_name: str
    # 模型名称。
    model_name: str
    # 节点自身输入 token 数。
    prompt_tokens: int
    # 节点自身输出 token 数。
    completion_tokens: int
    # 节点自身 total token 数。
    total_tokens: int


class TokenUsageState(TypedDict):
    """消息级 token 使用汇总。"""

    # 消息输入 token 总数（仅节点模型）。
    prompt_tokens: int
    # 消息输出 token 总数（仅节点模型）。
    completion_tokens: int
    # 消息 total token 总数（仅节点模型）。
    total_tokens: int
    # 是否所有 LLM 调用都拿到了 usage。
    is_complete: bool
    # 节点级 token 明细。
    node_breakdown: list[NodeTokenBreakdownState]


class GatewayRoutingState(TypedDict):
    """Gateway 路由结果结构。"""

    # 目标节点。
    route_target: str
    # 任务难度。
    task_difficulty: str


class AgentState(MessagesState, total=False):

    # Gateway 结构化路由结果。
    routing: GatewayRoutingState

    # 节点间共享结构化上下文。
    context: str

    # 对话历史消息。
    history_messages: list[ChatHistoryMessage]

    # 节点执行追踪（仅在 workflow 运行过程中暂存，流结束后统一落库）。
    execution_traces: list[ExecutionTraceState]

    # 消息级 token 汇总（与 execution_traces 分离，便于直接落库）。
    token_usage: TokenUsageState | None

    # 节点输出
    result: str
