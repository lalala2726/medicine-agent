from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.admin.agent_state import AgentState
from app.core.assistant_status import status_node
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


@status_node(node="chat", start_message="正在组织聊天内容")
@traceable(name="Chat Agent Node", run_type="chain")
def chat_agent(state: AgentState) -> AgentState:
    user_input = str(state.get("user_input") or "").strip()
    if not user_input:
        user_input = "你好"

    try:
        llm = create_chat_model(
            temperature=1.3,
            model="qwen-flash",
        )
        messages = [
            SystemMessage(content=_CHAT_SYSTEM_PROMPT),
            HumanMessage(content=user_input),
        ]
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
        "mode": "chat",
        "content": content,
    }
    return {"results": results}
