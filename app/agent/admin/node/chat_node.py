from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from app.agent.admin.state import AgentState
from app.core.langsmith import traceable
from app.schemas.prompt import base_prompt

_CHAT_SYSTEM_PROMPT = (
        """
            你是药品商城后台管理助手中的聊天节点（chat_agent）。
            你只处理闲聊、寒暄、通用说明，不负责订单/商品结果汇总。
            
            回复规则：
            1. 简洁、礼貌、自然，不要重复句子。
            2. 不要输出“我将调用工具”或内部调度细节。
            3. 若用户明显是业务查询（订单/商品/表格），给一句简短引导即可，不臆测数据。
    """
        + base_prompt
)


@traceable(name="Supervisor Chat Agent Node", run_type="chain")
def chat_agent(state: AgentState) -> dict[str, Any]:
    user_input = str(state.get("user_input") or "").strip()
    content = f"收到，你这条是闲聊内容：{user_input}" if user_input else "你好，这里是闲聊节点。"

    routing = dict(state.get("routing") or {})
    routing["mode"] = "chat"

    return {
        "result": content,
        "routing": routing,
        "messages": [AIMessage(content=content)],
    }
