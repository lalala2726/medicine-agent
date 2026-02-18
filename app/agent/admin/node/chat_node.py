import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.admin.agent_utils import build_execution_trace_update
from app.agent.admin.agent_state import AgentState
from app.agent.admin.history_utils import build_messages_with_history
from app.core.langsmith import traceable
from app.core.llm import create_chat_model
from app.schemas.prompt import base_prompt
from app.utils.streaming_utils import extract_text, invoke

_CHAT_SYSTEM_PROMPT = (
        """
    你是药品商城后台管理助手里的聊天节点（chat_agent）。
    你的职责是处理非业务问题，例如日常问候、使用方式说明、通用建议等轻量对话。
    
    约束：
    1. 不编造订单、售后、库存、报表等业务数据。
    2. 若用户问题明显属于业务查询/处理，请礼貌提示该问题应走业务节点。
    3. 回复尽量简洁、自然、直接。
    """
        + base_prompt
)

_FALLBACK_CHAT_SYSTEM_PROMPT = (
        """
    你是药品商城后台管理助手里的兜底解释节点（chat_agent fallback mode）。
    你将收到一个调度失败上下文（包含失败步骤与部分成功结果），你的输出目标是：
    1. 先说明本次未能完整执行的核心原因（简洁、直接，不使用技术术语堆砌）；
    2. 再给出已成功完成的部分结果摘要；
    3. 如果失败原因里包含具体步骤和错误信息，要用用户可理解的话转述；
    4. 不要编造任何不存在的数据。
    """
        + base_prompt
)


def _serialize_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """
    序列化节点输入消息，供 execution_trace 存储。

    Args:
        messages: 待序列化的消息列表。

    Returns:
        list[dict[str, Any]]: 统一结构的消息字典列表。
    """
    return [
        {
            "role": str(getattr(message, "type", "") or message.__class__.__name__).strip().lower() or "unknown",
            "content": getattr(message, "content", ""),
        }
        for message in messages
    ]


@traceable(name="Chat Agent Node", run_type="chain")
def chat_agent(state: AgentState) -> AgentState:
    """
    执行聊天节点（含 fallback 聊天模式）。

    Args:
        state: 当前全局状态，包含用户输入、历史对话和 fallback 上下文。

    Returns:
        AgentState: 增量更新结果，包含：
            - `results["chat"]`
            - `execution_traces`（记录节点输入输出）
    """
    routing = state.get("routing") or {}
    fallback_context = routing.get("fallback_context")
    is_fallback = isinstance(fallback_context, dict) and bool(fallback_context)

    user_input = str(state.get("user_input") or "").strip()
    if not user_input:
        user_input = "你好"
    history_messages = list(state.get("history_messages") or [])

    model_name = "qwen-flash"
    messages: list[Any] = []
    try:
        llm = create_chat_model(
            temperature=1.3,
            model=model_name,
        )
        if is_fallback:
            fallback_input = json.dumps(fallback_context, ensure_ascii=False)
            messages = [
                SystemMessage(content=_FALLBACK_CHAT_SYSTEM_PROMPT),
                HumanMessage(content=fallback_input),
            ]
        else:
            messages = build_messages_with_history(
                system_prompt=_CHAT_SYSTEM_PROMPT,
                history_messages=history_messages,
                fallback_user_input=user_input,
            )
        if hasattr(llm, "stream") and callable(getattr(llm, "stream")):
            streamed_parts: list[str] = []
            for chunk in llm.stream(messages):
                part = extract_text(chunk)
                if part:
                    streamed_parts.append(part)
            content = "".join(streamed_parts)
            if not content:
                content = invoke(llm, messages)
        else:
            content = invoke(llm, messages)
    except Exception:
        content = "聊天服务暂时不可用，请稍后重试。"

    results = dict(state.get("results") or {})
    results["chat"] = {
        "mode": "fallback" if is_fallback else "chat",
        "content": content,
    }
    result_update: dict[str, Any] = {"results": results}
    result_update.update(
        build_execution_trace_update(
            node_name="chat_agent",
            model_name=model_name,
            input_messages=_serialize_messages(messages),
            output_text=content,
            tool_calls=[],
        )
    )
    return result_update
