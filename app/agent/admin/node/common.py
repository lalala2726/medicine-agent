from __future__ import annotations

import json
from typing import Any, Iterable

from langchain_core.messages import AIMessage, BaseMessage

from agent.admin.state import AgentState
from app.agent.admin.agent_utils import build_execution_trace_update


def serialize_messages(messages: Iterable[BaseMessage]) -> list[dict[str, Any]]:
    """
    Serialize LangChain messages for execution traces.
    """
    serialized: list[dict[str, Any]] = []
    for message in messages:
        serialized.append(
            {
                "role": str(getattr(message, "type", "") or message.__class__.__name__).strip().lower() or "unknown",
                "content": getattr(message, "content", ""),
            }
        )
    return serialized


def _append_unique(items: list[str], values: Iterable[str]) -> list[str]:
    seen = set(items)
    for value in values:
        candidate = str(value).strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        items.append(candidate)
    return items


def _normalize_key(key: str) -> str:
    return "".join(ch for ch in key.strip().lower() if ch.isalnum())


def _extract_scalar_ids(raw: Any) -> list[str]:
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
    Extract IDs from tool outputs in diagnostics details.
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
    Merge node output payload into shared context.
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
    Build standardized worker node update payload.
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

