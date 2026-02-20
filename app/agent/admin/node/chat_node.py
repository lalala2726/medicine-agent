from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from app.agent.admin.agent_utils import build_execution_trace_update
from app.agent.admin.history_utils import build_messages_with_history
from app.agent.admin.node.common import serialize_messages
from app.agent.admin.state import AgentState
from app.core.langsmith import traceable
from app.core.llm import create_chat_model
from app.schemas.prompt import base_prompt
from app.utils.streaming_utils import extract_text, invoke

_CHAT_SYSTEM_PROMPT = (
        """
你是药品商城后台管理助手中的聊天节点（chat_agent）。
只处理闲聊、寒暄、通用说明，不编造业务数据。
若用户问题明显是订单/商品业务，简短提示将由业务节点继续处理。
"""
        + base_prompt
)


@traceable(name="Supervisor Chat Agent Node", run_type="chain")
def chat_agent(state: AgentState) -> dict[str, Any]:
    """
    Execute chat node.
    """
    user_input = str(state.get("user_input") or "").strip() or "你好"
    messages = build_messages_with_history(
        system_prompt=_CHAT_SYSTEM_PROMPT,
        history_messages=list(state.get("messages") or []),
        fallback_user_input=user_input,
    )

    model_name = "qwen-flash"
    content = "聊天服务暂时不可用，请稍后重试。"
    try:
        llm = create_chat_model(
            model=model_name,
            temperature=1.0,
        )
        if hasattr(llm, "stream") and callable(getattr(llm, "stream")):
            parts: list[str] = []
            for chunk in llm.stream(messages):
                part = extract_text(chunk)
                if part:
                    parts.append(part)
            content = "".join(parts) or invoke(llm, messages)
        else:
            content = invoke(llm, messages)
    except Exception:
        pass

    results = dict(state.get("results") or {})
    results["chat"] = {
        "mode": "chat",
        "content": content,
        "is_end": True,
    }
    update: dict[str, Any] = {
        "results": results,
        "messages": [AIMessage(content=content)],
        "context": {
            "last_agent": "chat_agent",
            "last_agent_response": content,
        },
    }
    update.update(
        build_execution_trace_update(
            node_name="chat_agent",
            model_name=model_name,
            input_messages=serialize_messages(messages),
            output_text=content,
            tool_calls=[],
        )
    )
    return update
