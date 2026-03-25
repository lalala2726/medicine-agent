from __future__ import annotations

from typing import Any, TypeAlias, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import MessagesState

ChatHistoryMessage: TypeAlias = HumanMessage | AIMessage | SystemMessage
"""对话历史消息类型别名。"""


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
    # 工具调用 ID。
    tool_call_id: str | None
    # 工具输入参数。
    tool_input: Any


class ExecutionTraceState(TypedDict, total=False):
    """单个节点执行追踪结构。"""

    # 节点执行顺序（从 1 开始）。
    sequence: int
    # 节点名称（当前 admin 工作流固定为 `admin_agent`）。
    node_name: str
    # 节点调用的模型名称。
    model_name: str
    # 节点状态（success/error）。
    status: str
    # 节点输出文本。
    output_text: str
    # 节点 LLM usage 是否完整。
    llm_usage_complete: bool
    # 节点自身 LLM token 使用量。
    llm_token_usage: TokenCounterState | None
    # 节点下发生的工具调用轨迹。
    tool_calls: list[ToolCallTraceState]
    # 节点扩展上下文（例如最终已授权工具数组）。
    node_context: dict[str, Any] | None


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


class AgentState(MessagesState, total=False):
    """
    管理端单 Agent 工作流状态。

    字段说明：
    1. `messages` 由 `MessagesState` 提供，兼容 LangGraph 内部消息流；
    2. `conversation_uuid` 用于会话级工具缓存隔离；
    3. `history_messages` 存储外层会话历史；
    4. `granted_tool_keys` 用于记录当前一次运行中已授权的业务工具；
    5. `execution_traces/token_usage/result` 用于外层持久化与流式落库。
    """

    # 当前会话 UUID。
    conversation_uuid: str

    # 对话历史消息。
    history_messages: list[ChatHistoryMessage]

    # 当前一次运行中已授权的业务工具 key 数组。
    granted_tool_keys: list[str]

    # 节点执行追踪（仅在 workflow 运行过程中暂存，流结束后统一落库）。
    execution_traces: list[ExecutionTraceState]

    # 消息级 token 汇总（与 execution_traces 分离，便于直接落库）。
    token_usage: TokenUsageState | None

    # 节点最终输出文本。
    result: str
