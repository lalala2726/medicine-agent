from __future__ import annotations

from typing import Annotated, Any, TypeAlias

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import MessagesState

# 对话历史消息类型：仅允许用户与助手消息，兼容当前落库与回放逻辑。
ChatHistoryMessage: TypeAlias = HumanMessage | AIMessage


def _merge_dict(
        left: dict[str, Any] | None,
        right: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    LangGraph reducer：合并两个字典状态片段。

    Args:
        left: 旧状态中的字典值，可为空。
        right: 新增量状态中的字典值，可为空。

    Returns:
        dict[str, Any]: 合并后的新字典；以 `right` 同名键覆盖 `left`。
    """

    merged = dict(left or {})
    merged.update(right or {})
    return merged


def _merge_list(
        left: list[Any] | None,
        right: list[Any] | None,
) -> list[Any]:
    """
    LangGraph reducer：拼接两个列表状态片段。

    Args:
        left: 旧状态中的列表值，可为空。
        right: 新增量状态中的列表值，可为空。

    Returns:
        list[Any]: 拼接后的新列表，顺序为 `left + right`。
    """

    return list(left or []) + list(right or [])


class AgentState(MessagesState, total=False):
    """
    Gateway + Supervisor 工作流全局状态定义。

    Attributes:
        user_input: 当前轮用户输入文本。
        next_node: supervisor 下发的下一跳节点名，或 `FINISH`。
        context: 跨节点共享上下文（如提取出的订单/商品 ID、节点输出缓存等）。
        routing: 路由元数据（如 `mode`、`route_target`、`turn`、`directive`、`finished`）。
        history_messages: 兼容现有服务层的历史字段（迁移期保留）。
        results: 节点输出聚合结果字典。
        execution_traces: 节点执行追踪列表（输入消息、工具调用、输出文本等）。
        errors: 错误信息列表。
    """

    user_input: str
    next_node: str

    # 节点间共享结构化上下文。
    context: Annotated[dict[str, Any], _merge_dict]

    # 路由相关元数据，包含 mode/route_target/turn/directive/finished 等字段。
    routing: Annotated[dict[str, Any], _merge_dict]

    # 保持历史字段以兼容当前服务层落库流程。
    history_messages: list[ChatHistoryMessage]

    results: Annotated[dict[str, Any], _merge_dict]
    execution_traces: Annotated[list[dict[str, Any]], _merge_list]
    errors: Annotated[list[str], _merge_list]
