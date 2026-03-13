from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.rag.query.constants import RAG_RANKING_PROMPT
from app.rag.query.runtime import KnowledgeSearchRuntimeConfig
from app.rag.query.types import KnowledgeSearchHit
from app.rag.query.utils import extract_message_content_text, strip_markdown_json_fence


def build_ranking_request_payload(
        *,
        query: str,
        hits: list[KnowledgeSearchHit],
        top_n: int,
) -> str:
    """构造排序模型使用的 JSON 输入文本。

    Args:
        query: 用户原始问题。
        hits: 当前候选知识片段列表。
        top_n: 期望保留的命中数量。

    Returns:
        传给排序模型的 JSON 字符串。
    """

    payload = {
        "query": query,
        "top_n": top_n,
        "documents": [
            {
                "serial_no": index,
                "knowledge_name": hit.knowledge_name,
                "content": hit.content,
            }
            for index, hit in enumerate(hits, start=1)
        ],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def parse_ranking_serial_numbers(*, response_text: str, max_serial_no: int) -> list[int]:
    """解析排序模型返回的目标序号列表。

    Args:
        response_text: 排序模型的原始响应文本。
        max_serial_no: 当前候选列表允许的最大序号。

    Returns:
        保序、去重、过滤越界后的有效序号列表。

    Raises:
        ValueError: 当响应不是合法 JSON 或缺少有效序号时抛出。
    """

    normalized_text = strip_markdown_json_fence(response_text)
    payload = json.loads(normalized_text)
    if not isinstance(payload, dict):
        raise ValueError("排序模型返回的 JSON 根节点必须是对象")

    raw_numbers = payload.get("top_serial_numbers")
    if not isinstance(raw_numbers, list):
        raise ValueError("排序模型返回缺少 top_serial_numbers 数组")

    serial_numbers: list[int] = []
    seen_numbers: set[int] = set()
    for item in raw_numbers:
        try:
            resolved = int(item)
        except (TypeError, ValueError):
            continue
        if resolved <= 0 or resolved > max_serial_no or resolved in seen_numbers:
            continue
        seen_numbers.add(resolved)
        serial_numbers.append(resolved)
    if not serial_numbers:
        raise ValueError("排序模型未返回可用序号")
    return serial_numbers


def rank_hits_with_chat_model(
        *,
        query: str,
        hits: list[KnowledgeSearchHit],
        runtime_config: KnowledgeSearchRuntimeConfig,
        final_top_k: int,
        build_ranking_chat_model: Callable[..., Any],
) -> list[KnowledgeSearchHit]:
    """使用普通聊天模型对候选知识片段进行排序。

    Args:
        query: 排序阶段使用的用户问题。
        hits: 待排序候选片段列表。
        runtime_config: 当前知识检索运行时配置。
        final_top_k: 最终期望返回的命中数量。
        build_ranking_chat_model: 排序模型构造函数。

    Returns:
        排序后的命中列表；当模型返回不足时自动用向量排序结果补齐。
    """

    ranking_model = build_ranking_chat_model(runtime_config=runtime_config)
    response = ranking_model.invoke(
        [
            SystemMessage(content=RAG_RANKING_PROMPT),
            HumanMessage(
                content=build_ranking_request_payload(
                    query=query,
                    hits=hits,
                    top_n=final_top_k,
                )
            ),
        ]
    )
    response_text = extract_message_content_text(getattr(response, "content", ""))
    serial_numbers = parse_ranking_serial_numbers(
        response_text=response_text,
        max_serial_no=len(hits),
    )

    ranked_hits: list[KnowledgeSearchHit] = []
    selected_indexes: set[int] = set()
    for serial_no in serial_numbers:
        index = serial_no - 1
        selected_indexes.add(index)
        ranked_hits.append(hits[index])
        if len(ranked_hits) >= final_top_k:
            return ranked_hits

    for index, hit in enumerate(hits):
        if index in selected_indexes:
            continue
        ranked_hits.append(hit)
        if len(ranked_hits) >= final_top_k:
            break
    return ranked_hits
