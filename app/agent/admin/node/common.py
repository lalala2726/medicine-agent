from __future__ import annotations

import json
from typing import Any, Iterable

from langchain_core.messages import AIMessage

from app.agent.admin.agent_utils import build_execution_trace_update, serialize_messages
from app.agent.admin.state import AgentState


def _append_unique(items: list[str], values: Iterable[str]) -> list[str]:
    """
    将去重后的新值追加到已有列表末尾。

    Args:
        items: 已有字符串列表，会在原列表基础上就地追加。
        values: 待追加的候选值序列，会自动进行 `str().strip()` 标准化。

    Returns:
        list[str]: 追加后的原列表对象。仅当候选值非空且此前不存在时才会写入。
    """

    seen = set(items)
    for value in values:
        candidate = str(value).strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        items.append(candidate)
    return items


def _normalize_key(key: str) -> str:
    """
    将字典键名归一化为仅含字母数字的小写形式。

    Args:
        key: 原始键名字符串。

    Returns:
        str: 去除空白并删除非字母数字字符后的键名，可用于跨接口字段名匹配。
    """

    return "".join(ch for ch in key.strip().lower() if ch.isalnum())


def _extract_scalar_ids(raw: Any) -> list[str]:
    """
    从任意输入中提取可作为 ID 的标量值。

    Args:
        raw: 任意输入，支持 `str`、`int`、`list` 递归结构。

    Returns:
        list[str]: 提取到的 ID 字符串列表。
            - `None` 返回空列表；
            - `str/int` 返回单元素列表；
            - `list` 递归展开后返回聚合结果；
            - 其他类型返回空列表。
    """

    if raw is None:
        return []
    if isinstance(raw, (str, int)):
        return [str(raw)]
    if isinstance(raw, list):
        values: list[str] = []
        for item in raw:
            values.extend(_extract_scalar_ids(item))
        return values
    return []


def _collect_ids_by_target(
        payload: Any,
        *,
        target: str,
        path: list[str] | None = None,
) -> list[str]:
    """
    从嵌套工具输出中按目标类型递归提取 ID。

    Args:
        payload: 工具输出对象，支持 `dict/list` 的任意嵌套。
        target: 目标类型，仅支持 `order` 或 `product`。
        path: 当前递归路径（归一化后的键名列表），内部递归使用，外部无需传入。

    Returns:
        list[str]: 命中目标字段规则后提取到的 ID 列表，未命中时返回空列表。
    """

    current_path = list(path or [])
    collected: list[str] = []

    if isinstance(payload, dict):
        for raw_key, value in payload.items():
            key = str(raw_key)
            normalized = _normalize_key(key)
            next_path = [*current_path, normalized]

            is_target_key = False
            if target == "order":
                is_target_key = (
                    normalized in {"orderid", "orderids"}
                    or ("order" in normalized and normalized.endswith("id"))
                    or (
                            normalized == "id"
                            and any("order" in segment for segment in current_path)
                    )
                )
            elif target == "product":
                is_target_key = (
                    normalized in {"productid", "productids", "drugid", "drugids"}
                    or (
                            (
                                    "product" in normalized
                                    or "drug" in normalized
                            )
                            and normalized.endswith("id")
                    )
                    or (
                            normalized == "id"
                            and any(
                        ("product" in segment or "drug" in segment)
                        for segment in current_path
                    )
                    )
                )

            if is_target_key:
                collected.extend(_extract_scalar_ids(value))

            collected.extend(
                _collect_ids_by_target(
                    value,
                    target=target,
                    path=next_path,
                )
            )
        return collected

    if isinstance(payload, list):
        for item in payload:
            collected.extend(
                _collect_ids_by_target(
                    item,
                    target=target,
                    path=current_path,
                )
            )
        return collected

    return collected


def extract_ids_from_tool_calls(
        tool_calls: list[dict[str, Any]],
        *,
        target: str,
) -> list[str]:
    """
    从工具调用明细中提取订单或商品 ID。

    Args:
        tool_calls: `execute_tool_node` 产出的工具调用明细列表。
            每个元素通常包含 `tool_output` 字段。
        target: 提取目标，支持 `order` 或 `product`。

    Returns:
        list[str]: 去重后的 ID 列表，保持首次出现顺序。
    """

    ids: list[str] = []
    for detail in tool_calls:
        if not isinstance(detail, dict):
            continue
        output = detail.get("tool_output")
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except Exception:
                pass
        extracted = _collect_ids_by_target(output, target=target)
        ids = _append_unique(ids, extracted)
    return ids


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
    将当前 worker 的执行结果合并进共享 context。

    Args:
        context: 现有上下文字典。
        node_name: 当前节点名称，如 `order_agent`。
        content: 节点对用户输出文本。
        status: 节点执行状态，常见值为 `completed` 或 `failed`。
        tool_calls: 工具调用明细列表。
        error: 节点错误信息，成功时可为 `None`。

    Returns:
        dict[str, Any]: 合并后的新上下文，关键字段说明：
            - `agent_outputs[node_name]`: 当前节点执行快照；
            - `last_agent` / `last_agent_response`: 最近一次节点与其输出；
            - `extracted_order_ids` / `extracted_product_ids`: 从工具输出累计提取的 ID。
    """

    merged_context = dict(context)
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

    if node_name == "order_agent":
        existing_order_ids = list(merged_context.get("extracted_order_ids") or [])
        existing_product_ids = list(merged_context.get("extracted_product_ids") or [])
        merged_context["extracted_order_ids"] = _append_unique(
            existing_order_ids,
            extract_ids_from_tool_calls(tool_calls, target="order"),
        )
        merged_context["extracted_product_ids"] = _append_unique(
            existing_product_ids,
            extract_ids_from_tool_calls(tool_calls, target="product"),
        )

    if node_name == "product_agent":
        existing_product_ids = list(merged_context.get("extracted_product_ids") or [])
        merged_context["extracted_product_ids"] = _append_unique(
            existing_product_ids,
            extract_ids_from_tool_calls(tool_calls, target="product"),
        )

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
        result_key: 写入 `results` 的目标键（如 `order`）。
        content: 节点输出文本。
        status: 节点执行状态。
        model_name: 执行模型名。
        input_messages: 模型输入消息（已序列化结构）。
        tool_calls: 工具调用明细。
        error: 错误信息；成功时可为空。

    Returns:
        dict[str, Any]: 标准化更新字典，包含：
            - `results[result_key]`：输出内容与 `is_end`；
            - `context`：合并后的共享上下文；
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
    "extract_ids_from_tool_calls",
    "merge_context_outputs",
    "serialize_messages",
]
