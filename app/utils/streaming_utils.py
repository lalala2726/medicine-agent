from __future__ import annotations

from typing import Any

from app.agent.admin.agent_state import AgentState


def has_plan(state: AgentState | dict[str, Any]) -> bool:
    raw_plan = state.get("plan")
    return isinstance(raw_plan, list) and len(raw_plan) > 0


def should_stream_node_output(state: AgentState | dict[str, Any], node_name: str) -> bool:
    """
    系统自动判断节点是否是最终输出节点（无需依赖模型字段）：
    1. gateway_router 直达该节点且无 plan。
    2. planner 标记最后阶段，且 next_nodes 仅包含该节点。
    """
    routing = state.get("routing") or {}
    route_target = routing.get("route_target")
    if route_target == node_name and not has_plan(state):
        return True

    next_nodes = routing.get("next_nodes")
    return (
        bool(routing.get("is_final_stage"))
        and isinstance(next_nodes, list)
        and len(next_nodes) == 1
        and next_nodes[0] == node_name
    )


def message_chunk_to_text(message_chunk: Any) -> str:
    content = getattr(message_chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
            elif isinstance(item, str):
                text_parts.append(item)
        return "".join(text_parts)
    if content is None:
        return ""
    return str(content)


def generate_content_with_stream_fallback(llm: Any, messages: list[Any], enable_stream: bool) -> tuple[str, list[str]]:
    """
    统一的节点生成策略：
    - enable_stream=True 时优先 stream，失败则回退 invoke。
    - enable_stream=False 时直接 invoke。
    """
    if enable_stream:
        try:
            stream_chunks: list[str] = []
            for chunk in llm.stream(messages):
                chunk_text = message_chunk_to_text(chunk)
                if chunk_text:
                    stream_chunks.append(chunk_text)
            if stream_chunks:
                return "".join(stream_chunks), stream_chunks
        except Exception:
            pass

    response = llm.invoke(messages)
    return str(response.content), []
