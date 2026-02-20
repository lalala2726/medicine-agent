from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from app.agent.admin.agent_utils import build_execution_trace_update, serialize_messages
from app.agent.admin.history_utils import build_messages_with_history
from app.agent.admin.model_policy import normalize_task_difficulty, resolve_model_profile
from app.agent.admin.state import AgentState
from app.core.assistant_status import emit_thinking_notice
from app.core.langsmith import traceable
from app.core.llm import create_chat_model
from app.schemas.prompt import base_prompt
from app.utils.streaming_utils import invoke, stream_with_reasoning

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
    """
    执行纯闲聊节点并返回聊天结果。

    Args:
        state: 当前图状态，函数读取 `messages` 与 `user_input` 组装聊天输入。

    Returns:
        dict[str, Any]: 聊天节点状态增量，核心字段如下：
            - `results.chat`: 闲聊输出（`mode=chat`，`is_end=True`）；
            - `messages`: 新增一条 AIMessage；
            - `context.last_agent/last_agent_response`: 最近输出节点信息；
            - `execution_traces`: 输入与输出追踪。
    """

    user_input = str(state.get("user_input") or "").strip() or "你好"
    routing = dict(state.get("routing") or {})
    task_difficulty = normalize_task_difficulty(routing.get("task_difficulty"))
    model_profile = resolve_model_profile(task_difficulty)

    messages = build_messages_with_history(
        system_prompt=_CHAT_SYSTEM_PROMPT,
        history_messages=list(state.get("messages") or []),
        fallback_user_input=user_input,
    )

    model_name = str(model_profile.get("model") or "qwen-plus")
    think_enabled = bool(model_profile.get("think"))
    content = "聊天服务暂时不可用，请稍后重试。"
    try:
        llm = create_chat_model(
            model=model_name,
            think=think_enabled,
            temperature=1.0,
        )
        if hasattr(llm, "stream") and callable(getattr(llm, "stream")):
            answer_chunks, reasoning_chunks = stream_with_reasoning(llm, messages)
            if think_enabled and reasoning_chunks:
                emit_thinking_notice(
                    node="chat_agent",
                    state="thinking_start",
                    meta={
                        "model": model_name,
                        "task_difficulty": task_difficulty,
                    },
                )
                for chunk in reasoning_chunks:
                    emit_thinking_notice(
                        node="chat_agent",
                        state="thinking_delta",
                        text=chunk,
                        meta={
                            "model": model_name,
                            "task_difficulty": task_difficulty,
                        },
                    )
                emit_thinking_notice(
                    node="chat_agent",
                    state="thinking_end",
                    meta={
                        "model": model_name,
                        "task_difficulty": task_difficulty,
                    },
                )
            content = "".join(answer_chunks) or invoke(llm, messages)
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
