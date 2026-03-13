from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pymilvus import MilvusClient

from app.core.codes import ResponseCode
from app.core.config_sync import get_current_agent_config_snapshot
from app.core.exception.exceptions import ServiceException
from app.core.llms import create_chat_model, create_embedding_model
from app.repositories import vector_repository
from app.rag.query.types import KnowledgeSearchHit
from app.utils.prompt_utils import load_prompt

#: 知识库切片检索时只命中启用状态数据。
RAG_FILTER_EXPR = "status == 0"
#: 复用项目当前 Milvus 向量索引的默认检索参数。
RAG_SEARCH_PARAMS = {"metric_type": vector_repository.DEFAULT_VECTOR_METRIC_TYPE, "params": {}}
#: 知识检索返回给查询层的标准输出字段。
RAG_OUTPUT_FIELDS = ["document_id", "chunk_index", "char_count", "content"]
#: 当显式参数与 Redis 都未提供时的默认最终返回条数。
RAG_DEFAULT_FINAL_TOP_K = 10
#: 最终返回条数允许的最大值，避免上下文被无限放大。
RAG_MAX_FINAL_TOP_K = 100
#: 输出给 Agent 的知识上下文最大字符预算。
RAG_MAX_CONTEXT_CHARS = 12000
#: 单次最多允许同时查询的知识库数量。
RAG_MAX_KNOWLEDGE_NAMES = 10
#: 启用排序时的候选池最大规模。
RAG_MAX_CANDIDATE_POOL = 100
#: 排序模型的固定最大输出 token。
RAG_RANKING_MAX_TOKENS = 512
#: 要求底层聊天模型尽量返回合法 JSON。
RAG_RANKING_RESPONSE_FORMAT = {"response_format": {"type": "json_object"}}
#: 检索问题改写链路使用的系统提示词。
_RAG_REWRITE_PROMPT = load_prompt("_system/rewrite_rag_query.md").strip()
#: 知识片段排序链路使用的系统提示词。
_RAG_RANKING_PROMPT = """
    你是知识库结果排序器。你的任务是根据用户问题，从候选文档中选出最能直接回答问题的前 top_n 个片段。
    
    必须遵守：
    1. 仅输出一个 JSON 对象。
    2. JSON 结构固定为 {"top_serial_numbers":[整数序号,...]}。
    3. 只能返回候选 documents 中已经出现过的 serial_no。
    4. 只返回前 top_n 个序号，不要解释，不要附加 markdown，不要输出其他字段。
    5. 优先选择最能回答问题的片段，而不是仅仅主题相近的片段。
    6. 如果候选里没有足够合适的片段，也只返回你认为最相关的序号。
""".strip()
#: 控制 RAG 运行时配置日志仅打印一次，避免重复刷屏。
_RAG_RUNTIME_CONFIG_LOGGED = False


@dataclass(frozen=True)
class _KnowledgeSearchRuntimeConfig:
    """知识检索运行时配置。"""

    provider_type: str
    llm_base_url: str
    llm_api_key: str
    knowledge_names: list[str]
    embedding_model_name: str
    embedding_dim: int
    ranking_enabled: bool
    ranking_model_name: str | None
    configured_top_k: int | None


def _normalize_question(question: str) -> str:
    """规范化并校验知识库检索问题。

    Args:
        question: 用户传入的原始问题文本。

    Returns:
        去除首尾空白后的问题文本。

    Raises:
        ServiceException: 当问题在去空白后为空时抛出。
    """

    normalized_question = str(question or "").strip()
    if not normalized_question:
        raise ServiceException(
            code=ResponseCode.BAD_REQUEST,
            message="question 不能为空",
        )
    return normalized_question


def _normalize_top_k(top_k: int | None) -> int | None:
    """规范化显式传入的最终返回条数。

    Args:
        top_k: 调用方期望返回的命中数量。

    Returns:
        校验通过后的正整数；未传时返回 ``None``。

    Raises:
        ServiceException: 当 ``top_k`` 超出允许范围时抛出。
    """

    if top_k is None:
        return None
    try:
        normalized_top_k = int(top_k)
    except (TypeError, ValueError) as exc:
        raise ServiceException(
            code=ResponseCode.BAD_REQUEST,
            message="top_k 必须是 1 到 100 之间的整数",
        ) from exc
    if normalized_top_k <= 0 or normalized_top_k > RAG_MAX_FINAL_TOP_K:
        raise ServiceException(
            code=ResponseCode.BAD_REQUEST,
            message="top_k 必须在 1 到 100 之间",
        )
    return normalized_top_k


def _resolve_final_top_k(explicit_top_k: int | None) -> int:
    """解析当前请求最终生效的返回条数。

    Args:
        explicit_top_k: 调用方显式传入的最终返回条数。

    Returns:
        实际生效的最终返回条数。
    """

    normalized_explicit_top_k = _normalize_top_k(explicit_top_k)
    if normalized_explicit_top_k is not None:
        return normalized_explicit_top_k

    configured_top_k = get_current_agent_config_snapshot().get_knowledge_top_k()
    if configured_top_k is not None:
        return configured_top_k
    return RAG_DEFAULT_FINAL_TOP_K


def _coerce_optional_int(value: Any) -> int | None:
    """将元信息中的值规整为可选整数。"""

    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_message_content_text(content: Any) -> str:
    """从 LangChain 消息内容中提取纯文本。"""

    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    pieces: list[str] = []
    for item in content:
        if isinstance(item, str) and item.strip():
            pieces.append(item.strip())
            continue
        if isinstance(item, dict):
            candidate = item.get("text")
            if isinstance(candidate, str) and candidate.strip():
                pieces.append(candidate.strip())
    return "\n".join(pieces).strip()


def _strip_markdown_json_fence(text: str) -> str:
    """移除可能包裹 JSON 的 markdown 代码块围栏。"""

    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _resolve_runtime_config() -> _KnowledgeSearchRuntimeConfig:
    """解析当前知识检索链所需的运行时配置。"""

    snapshot = get_current_agent_config_snapshot()
    runtime_config = snapshot.get_llm_runtime_config()
    if runtime_config is None:
        raise ServiceException(
            code=ResponseCode.SERVICE_UNAVAILABLE,
            message="知识库检索缺少 llm 运行时配置",
        )
    if (
            runtime_config.provider_type is None
            or runtime_config.base_url is None
            or runtime_config.api_key is None
    ):
        raise ServiceException(
            code=ResponseCode.SERVICE_UNAVAILABLE,
            message="知识库检索的 llm 运行时配置不完整",
        )

    knowledge_names = snapshot.get_knowledge_names()
    if not knowledge_names:
        raise ServiceException(
            code=ResponseCode.SERVICE_UNAVAILABLE,
            message="知识库检索未配置 knowledgeNames",
        )
    if len(knowledge_names) > RAG_MAX_KNOWLEDGE_NAMES:
        raise ServiceException(
            code=ResponseCode.SERVICE_UNAVAILABLE,
            message=f"知识库数量不能超过 {RAG_MAX_KNOWLEDGE_NAMES} 个",
        )

    embedding_model_name = snapshot.get_knowledge_embedding_model_name()
    if embedding_model_name is None:
        raise ServiceException(
            code=ResponseCode.SERVICE_UNAVAILABLE,
            message="知识库检索未配置 embeddingModel",
        )

    embedding_dim = snapshot.get_knowledge_embedding_dim()
    if embedding_dim is None:
        raise ServiceException(
            code=ResponseCode.SERVICE_UNAVAILABLE,
            message="知识库检索未配置 embeddingDim",
        )

    return _KnowledgeSearchRuntimeConfig(
        provider_type=runtime_config.provider_type,
        llm_base_url=runtime_config.base_url,
        llm_api_key=runtime_config.api_key,
        knowledge_names=knowledge_names,
        embedding_model_name=embedding_model_name,
        embedding_dim=embedding_dim,
        ranking_enabled=snapshot.is_knowledge_ranking_enabled(),
        ranking_model_name=snapshot.get_knowledge_ranking_model_name(),
        configured_top_k=snapshot.get_knowledge_top_k(),
    )


def _build_rag_milvus_connection_args() -> dict[str, str | float]:
    """构造 RAG 查询专用的 Milvus 连接参数。"""

    from app.core.database.milvus import get_milvus_connection_args

    return dict(get_milvus_connection_args())


def _build_rag_embedding_client(*, runtime_config: _KnowledgeSearchRuntimeConfig) -> Any:
    """构造 RAG 查询专用的向量模型客户端。"""

    return create_embedding_model(
        provider=runtime_config.provider_type,
        model=runtime_config.embedding_model_name,
        api_key=runtime_config.llm_api_key,
        base_url=runtime_config.llm_base_url,
        dimensions=runtime_config.embedding_dim,
    )


def _build_rag_milvus_client(
        *,
        runtime_config: _KnowledgeSearchRuntimeConfig,
) -> MilvusClient:
    """构造 RAG 查询专用的 Milvus 客户端。"""

    connection_args = _build_rag_milvus_connection_args()
    _log_rag_runtime_config_once(
        runtime_config=runtime_config,
        connection_args=connection_args,
    )
    return MilvusClient(**connection_args)


def _build_ranking_chat_model(*, runtime_config: _KnowledgeSearchRuntimeConfig) -> Any:
    """构造知识片段排序使用的普通聊天模型。"""

    model_name = runtime_config.ranking_model_name
    if model_name is None:
        raise ServiceException(
            code=ResponseCode.SERVICE_UNAVAILABLE,
            message="知识库排序未配置 rankingModel",
        )
    return create_chat_model(
        model=model_name,
        provider=runtime_config.provider_type,
        api_key=runtime_config.llm_api_key,
        base_url=runtime_config.llm_base_url,
        extra_body=RAG_RANKING_RESPONSE_FORMAT,
        temperature=0.0,
        max_tokens=RAG_RANKING_MAX_TOKENS,
        think=False,
    )


def _log_rag_runtime_config_once(
        *,
        runtime_config: _KnowledgeSearchRuntimeConfig,
        connection_args: dict[str, str | float],
) -> None:
    """记录当前 RAG 查询链的关键运行时配置。"""

    global _RAG_RUNTIME_CONFIG_LOGGED
    if _RAG_RUNTIME_CONFIG_LOGGED:
        return

    logger.info(
        "RAG 查询配置已生效：provider={}，embedding_model={}，embedding_dimensions={}，"
        "llm_base_url={}，milvus_uri={}，milvus_db_name={}，knowledge_names={}，"
        "ranking_enabled={}，ranking_model={}，configured_top_k={}",
        runtime_config.provider_type,
        runtime_config.embedding_model_name,
        runtime_config.embedding_dim,
        runtime_config.llm_base_url,
        connection_args.get("uri"),
        connection_args.get("db_name"),
        runtime_config.knowledge_names,
        runtime_config.ranking_enabled,
        runtime_config.ranking_model_name,
        runtime_config.configured_top_k,
    )
    _RAG_RUNTIME_CONFIG_LOGGED = True


def _extract_search_row_value(row: dict[str, Any], field_name: str) -> Any:
    """从 Milvus 检索结果中提取字段值。"""

    if field_name in row:
        return row.get(field_name)
    entity = row.get("entity")
    if isinstance(entity, dict):
        return entity.get(field_name)
    return None


def _vector_score_from_row(row: dict[str, Any]) -> float:
    """从 Milvus 检索结果中提取当前用于排序的向量分数。"""

    return float(row.get("distance") or row.get("score") or 0.0)


def _search_collection_rows(
        *,
        client: MilvusClient,
        knowledge_name: str,
        query_vector: list[float],
        limit: int,
) -> list[dict[str, Any]] | None:
    """查询单个知识库 collection 并返回原始命中行。"""

    if not client.has_collection(knowledge_name):
        logger.warning("知识库集合不存在，已跳过：collection={}", knowledge_name)
        return None

    search_results = client.search(
        collection_name=knowledge_name,
        data=[query_vector],
        filter=RAG_FILTER_EXPR,
        limit=limit,
        output_fields=RAG_OUTPUT_FIELDS,
        search_params=RAG_SEARCH_PARAMS,
        anns_field="embedding",
    )
    if not search_results:
        return []
    return list(search_results[0] or [])


def _to_knowledge_hit(row: dict[str, Any], *, knowledge_name: str) -> KnowledgeSearchHit:
    """将 Milvus 检索结果行转换为统一的命中结构。"""

    return KnowledgeSearchHit(
        knowledge_name=knowledge_name,
        content=str(_extract_search_row_value(row, "content") or "").strip(),
        score=_vector_score_from_row(row),
        document_id=_coerce_optional_int(_extract_search_row_value(row, "document_id")),
        chunk_index=_coerce_optional_int(_extract_search_row_value(row, "chunk_index")),
        char_count=_coerce_optional_int(_extract_search_row_value(row, "char_count")),
    )


def _deduplicate_hits(hits: list[KnowledgeSearchHit]) -> list[KnowledgeSearchHit]:
    """按固定业务键去重，保留分数更高的命中。"""

    deduplicated: dict[tuple[str, int | None, int | None, str], KnowledgeSearchHit] = {}
    for hit in hits:
        dedupe_key = (
            hit.knowledge_name,
            hit.document_id,
            hit.chunk_index,
            hit.content,
        )
        existing = deduplicated.get(dedupe_key)
        if existing is None or hit.score > existing.score:
            deduplicated[dedupe_key] = hit
    return list(deduplicated.values())


def _sort_hits_desc(hits: list[KnowledgeSearchHit]) -> list[KnowledgeSearchHit]:
    """按分数从高到低排序。"""

    return sorted(hits, key=lambda item: item.score, reverse=True)


def _resolve_vector_candidate_target(*, final_top_k: int, ranking_enabled: bool) -> int:
    """解析当前阶段需要保留的向量候选池规模。"""

    if not ranking_enabled:
        return final_top_k
    return min(max(final_top_k * 3, final_top_k), RAG_MAX_CANDIDATE_POOL)


def _resolve_recall_per_kb(
        *,
        final_top_k: int,
        knowledge_count: int,
        ranking_enabled: bool,
) -> int:
    """根据最终返回条数与知识库数量解析每个知识库的召回规模。"""

    if knowledge_count <= 0:
        return 1
    candidate_target = _resolve_vector_candidate_target(
        final_top_k=final_top_k,
        ranking_enabled=ranking_enabled,
    )
    return max(1, math.ceil(candidate_target / knowledge_count))


def _search_knowledge_hits(
        *,
        question: str,
        final_top_k: int,
        runtime_config: _KnowledgeSearchRuntimeConfig,
        ranking_enabled: bool,
) -> list[KnowledgeSearchHit]:
    """执行多知识库聚合向量检索。"""

    client = _build_rag_milvus_client(runtime_config=runtime_config)
    embedding_client = _build_rag_embedding_client(runtime_config=runtime_config)
    query_vector = embedding_client.embed_query(question)
    recall_per_kb = _resolve_recall_per_kb(
        final_top_k=final_top_k,
        knowledge_count=len(runtime_config.knowledge_names),
        ranking_enabled=ranking_enabled,
    )
    candidate_target = _resolve_vector_candidate_target(
        final_top_k=final_top_k,
        ranking_enabled=ranking_enabled,
    )

    found_collections: list[str] = []
    aggregated_hits: list[KnowledgeSearchHit] = []
    for knowledge_name in runtime_config.knowledge_names:
        rows = _search_collection_rows(
            client=client,
            knowledge_name=knowledge_name,
            query_vector=query_vector,
            limit=recall_per_kb,
        )
        if rows is None:
            continue
        found_collections.append(knowledge_name)
        aggregated_hits.extend(
            _to_knowledge_hit(row, knowledge_name=knowledge_name)
            for row in rows
        )

    if not found_collections:
        raise ServiceException(
            code=ResponseCode.NOT_FOUND,
            message=f"知识库集合不存在: collections={runtime_config.knowledge_names}",
        )

    deduplicated_hits = _deduplicate_hits(aggregated_hits)
    return _sort_hits_desc(deduplicated_hits)[:candidate_target]


def _call_rewrite_llm(question: str) -> str:
    """使用聊天槽位模型改写检索问题。"""

    from app.core.config_sync import AgentChatModelSlot, create_agent_chat_llm

    llm_model = create_agent_chat_llm(
        slot=AgentChatModelSlot.CHAT,
        temperature=0.0,
        think=False,
    )
    response = llm_model.invoke(
        [
            SystemMessage(content=_RAG_REWRITE_PROMPT),
            HumanMessage(content=question),
        ]
    )
    return _extract_message_content_text(getattr(response, "content", ""))


def _rewrite_question_for_knowledge_search(question: str) -> str:
    """将原始问题改写为更适合向量检索的查询语句。"""

    normalized_question = _normalize_question(question)
    try:
        rewritten_question = _call_rewrite_llm(normalized_question)
    except Exception as exc:
        logger.opt(exception=exc).warning(
            "Failed to rewrite rag question, fallback to raw question.",
        )
        return normalized_question
    rewritten_question = rewritten_question.strip()
    return rewritten_question or normalized_question


def _build_ranking_request_payload(
        *,
        query: str,
        hits: list[KnowledgeSearchHit],
        top_n: int,
) -> str:
    """构造排序模型使用的 JSON 输入文本。"""

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


def _parse_ranking_serial_numbers(*, response_text: str, max_serial_no: int) -> list[int]:
    """解析排序模型返回的目标序号列表。"""

    normalized_text = _strip_markdown_json_fence(response_text)
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


def _rank_hits_with_chat_model(
        *,
        query: str,
        hits: list[KnowledgeSearchHit],
        runtime_config: _KnowledgeSearchRuntimeConfig,
        final_top_k: int,
) -> list[KnowledgeSearchHit]:
    """使用普通聊天模型对候选知识片段进行排序。"""

    ranking_model = _build_ranking_chat_model(runtime_config=runtime_config)
    response = ranking_model.invoke(
        [
            SystemMessage(content=_RAG_RANKING_PROMPT),
            HumanMessage(
                content=_build_ranking_request_payload(
                    query=query,
                    hits=hits,
                    top_n=final_top_k,
                )
            ),
        ]
    )
    response_text = _extract_message_content_text(getattr(response, "content", ""))
    serial_numbers = _parse_ranking_serial_numbers(
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


def _query_knowledge(
        *,
        vector_question: str,
        ranking_question: str,
        top_k: int | None,
) -> list[KnowledgeSearchHit]:
    """统一执行知识库查询与可选排序。"""

    final_top_k = _resolve_final_top_k(top_k)
    runtime_config = _resolve_runtime_config()

    vector_hits = _search_knowledge_hits(
        question=vector_question,
        final_top_k=final_top_k,
        runtime_config=runtime_config,
        ranking_enabled=runtime_config.ranking_enabled,
    )
    if not runtime_config.ranking_enabled:
        return vector_hits[:final_top_k]

    try:
        return _rank_hits_with_chat_model(
            query=ranking_question,
            hits=vector_hits,
            runtime_config=runtime_config,
            final_top_k=final_top_k,
        )[:final_top_k]
    except Exception as exc:
        logger.opt(exception=exc).warning(
            "知识库排序失败，已降级为向量排序：message={}",
            str(exc),
        )
        fallback_hits = _search_knowledge_hits(
            question=vector_question,
            final_top_k=final_top_k,
            runtime_config=runtime_config,
            ranking_enabled=False,
        )
        return fallback_hits[:final_top_k]


def query_knowledge_by_raw_question(
        *,
        question: str,
        top_k: int | None,
) -> list[KnowledgeSearchHit]:
    """使用原始问题直接检索知识库。"""

    normalized_question = _normalize_question(question)
    try:
        return _query_knowledge(
            vector_question=normalized_question,
            ranking_question=normalized_question,
            top_k=top_k,
        )
    except ServiceException:
        raise
    except Exception as exc:
        raise ServiceException(
            code=ResponseCode.OPERATION_FAILED,
            message=f"知识库检索失败: {exc}",
        ) from exc


def query_knowledge_by_rewritten_question(
        *,
        question: str,
        top_k: int | None,
) -> list[KnowledgeSearchHit]:
    """先改写问题，再检索知识库。"""

    normalized_question = _normalize_question(question)
    rewritten_question = _rewrite_question_for_knowledge_search(normalized_question)
    try:
        return _query_knowledge(
            vector_question=rewritten_question,
            ranking_question=normalized_question,
            top_k=top_k,
        )
    except ServiceException:
        raise
    except Exception as exc:
        raise ServiceException(
            code=ResponseCode.OPERATION_FAILED,
            message=f"知识库检索失败: {exc}",
        ) from exc


def _trim_content_to_budget(*, content: str, remaining_chars: int) -> str:
    """将片段内容裁剪到剩余预算以内。"""

    if remaining_chars <= 0:
        return ""
    if len(content) <= remaining_chars:
        return content
    if remaining_chars <= 3:
        return content[:remaining_chars]
    return f"{content[:remaining_chars - 3]}..."


def format_knowledge_search_hits(hits: list[KnowledgeSearchHit]) -> str:
    """将结构化命中结果渲染为 Agent 可消费的文本块。"""

    if not hits:
        return "未检索到相关知识。"

    lines = ["已检索到以下知识片段："]
    consumed_chars = 0
    rendered_count = 0
    for hit in hits:
        remaining_chars = RAG_MAX_CONTEXT_CHARS - consumed_chars
        if remaining_chars <= 0:
            break

        rendered_content = _trim_content_to_budget(
            content=hit.content,
            remaining_chars=remaining_chars,
        ).strip()
        if not rendered_content:
            break

        rendered_count += 1
        consumed_chars += len(rendered_content)
        meta_parts = [
            f"knowledge_name={hit.knowledge_name}",
            f"score={hit.score:.4f}",
        ]
        if hit.document_id is not None:
            meta_parts.append(f"document_id={hit.document_id}")
        if hit.chunk_index is not None:
            meta_parts.append(f"chunk_index={hit.chunk_index}")
        if hit.char_count is not None:
            meta_parts.append(f"char_count={hit.char_count}")
        lines.append(f"[片段{rendered_count}] {', '.join(meta_parts)}\n{rendered_content}")

    if rendered_count == 0:
        return "未检索到相关知识。"
    return "\n\n".join(lines)
