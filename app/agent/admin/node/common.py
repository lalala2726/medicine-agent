from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from app.agent.admin.agent_utils import build_execution_trace_update, serialize_messages
from app.agent.admin.state import AgentState


def merge_context_outputs(
        context: dict[str, Any],
        *,
        node_name: str,
        content: str,
        status: str,
        tool_calls: list[dict[str, Any]],
        error: str | None,
) -> dict[str, Any]:
    """
    将当前 worker 执行结果合并到共享上下文。

    设计约束：
    1. 不做任何固定字段提取（如 `extracted_order_ids` / `extracted_product_ids`）；
    2. 只保留节点原始输出快照，交给 supervisor 动态判断和传递；
    3. 保持上下文结构简单，降低硬编码耦合。

    Args:
        context: 现有上下文字典。
        node_name: 当前节点名称（例如 `order_agent`）。
        content: 节点输出文本。
        status: 节点执行状态，通常为 `completed` 或 `failed`。
        tool_calls: 当前节点工具调用明细列表。
        error: 错误信息，成功时为 `None`。

    Returns:
        dict[str, Any]: 合并后的上下文字典，包含：
            - `agent_outputs[node_name]`: 当前节点输出快照；
            - `last_agent`: 最近执行节点；
            - `last_agent_response`: 最近执行节点输出文本。
    """

    merged_context = dict(context)
    # 清理历史版本固定提取字段，避免继续耦合 `extracted_*` 约定。
    merged_context.pop("extracted_order_ids", None)
    merged_context.pop("extracted_product_ids", None)

    agent_outputs = dict(merged_context.get("agent_outputs") or {})
    agent_outputs[node_name] = {
        "status": status,
        "content": content,
        "tool_calls": tool_calls,
        "error": error,
    }
    merged_context["agent_outputs"] = agent_outputs
    merged_context["last_agent"] = node_name
    merged_context["last_agent_response"] = content
    return merged_context


def build_worker_update(
        *,
        state: AgentState,
        node_name: str,
        result_key: str,
        content: str,
        status: str,
        model_name: str,
        input_messages: list[Any],
        tool_calls: list[dict[str, Any]],
        error: str | None = None,
) -> dict[str, Any]:
    """
    生成统一的 worker 节点状态更新结构。

    Args:
        state: 当前图状态。
        node_name: 当前节点名。
        result_key: 写入 `results` 的目标键（如 `order`、`product`）。
        content: 节点输出文本。
        status: 节点执行状态。
        model_name: 执行模型名。
        input_messages: 模型输入消息（已序列化结构）。
        tool_calls: 工具调用明细。
        error: 错误信息；成功时可为空。

    Returns:
        dict[str, Any]: 标准化更新字典，包含：
            - `results[result_key]`：输出内容与 `is_end`；
            - `context`：仅追加节点原始输出；
            - `messages`：一条 `AIMessage`；
            - `execution_traces`：执行追踪；
            - `errors`（可选）：失败错误列表。
    """

    routing = dict(state.get("routing") or {})
    mode = str(routing.get("mode") or "")
    is_end = mode == "fast_lane"

    results = dict(state.get("results") or {})
    results[result_key] = {
        "content": content,
        "is_end": is_end,
    }

    context = merge_context_outputs(
        dict(state.get("context") or {}),
        node_name=node_name,
        content=content,
        status=status,
        tool_calls=tool_calls,
        error=error,
    )

    update: dict[str, Any] = {
        "results": results,
        "context": context,
        "messages": [AIMessage(content=content)],
    }
    if status == "failed" and error:
        update["errors"] = [error]

    update.update(
        build_execution_trace_update(
            node_name=node_name,
            model_name=model_name,
            input_messages=input_messages,
            output_text=content,
            tool_calls=tool_calls,
        )
    )
    return update


__all__ = [
    "build_worker_update",
    "merge_context_outputs",
    "serialize_messages",
]
