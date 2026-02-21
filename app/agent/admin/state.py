from __future__ import annotations

from typing import TypeAlias

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import MessagesState

ChatHistoryMessage: TypeAlias = HumanMessage | AIMessage


class AgentState(MessagesState, total=False):

    # 用户输入
    user_input: str

    # 节点路由
    router: str

    # 节点间共享结构化上下文。
    context: str

    # 对话历史消息。
    history_messages: list[ChatHistoryMessage]
