from __future__ import annotations

from typing import Any

from loguru import logger
from pymilvus import MilvusClient

from app.core.codes import ResponseCode
from app.core.database.milvus import get_milvus_connection_args
from app.core.exception.exceptions import ServiceException
from app.core.llms import create_chat_model, create_embedding_model
from app.rag.query import ranking as ranking_module
from app.rag.query import retriever as retriever_module
from app.rag.query import rewrite as rewrite_module
from app.rag.query import runtime as runtime_module
from app.rag.query.constants import (
    RAG_MAX_CONTEXT_CHARS,
    RAG_RANKING_MAX_TOKENS,
    RAG_RANKING_RESPONSE_FORMAT,
)
from app.rag.query.runtime import KnowledgeSearchRuntimeConfig
from app.rag.query.types import KnowledgeSearchHit
from app.rag.query.utils import normalize_question, trim_content_to_budget

#: 控制 RAG 运行时配置日志仅打印一次，避免重复刷屏。
_RAG_RUNTIME_CONFIG_LOGGED = False
#: 为兼容现有测试与内部调用保留的运行时配置别名。
_KnowledgeSearchRuntimeConfig = KnowledgeSearchRuntimeConfig


def _build_rag_embedding_client(*, runtime_config: KnowledgeSearchRuntimeConfig) -> Any:
    """构造 RAG 查询专用的向量模型客户端。

    Args:
        runtime_config: 当前请求生效的知识检索运行时配置。

    Returns:
        可执行 ``embed_query`` 的向量模型客户端实例。
    """

    return create_embedding_model(
        provider=runtime_config.provider_type,
        model=runtime_config.embedding_model_name,
        api_key=runtime_config.llm_api_key,
        base_url=runtime_config.llm_base_url,
        dimensions=runtime_config.embedding_dim,
    )


def _log_rag_runtime_config_once(
        *,
        runtime_config: KnowledgeSearchRuntimeConfig,
        connection_args: dict[str, str | float],
) -> None:
    """记录当前 RAG 查询链的关键运行时配置。

    Args:
        runtime_config: 当前请求生效的知识检索运行时配置。
        connection_args: 当前 Milvus 客户端使用的连接参数。

    Returns:
        ``None``。
    """

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


def _build_rag_milvus_client(
        *,
        runtime_config: KnowledgeSearchRuntimeConfig,
) -> MilvusClient:
    """构造 RAG 查询专用的 Milvus 客户端。

    Args:
        runtime_config: 当前请求生效的知识检索运行时配置。

    Returns:
        已绑定当前环境连接参数的 ``MilvusClient`` 实例。
    """

    connection_args = dict(get_milvus_connection_args())
    _log_rag_runtime_config_once(
        runtime_config=runtime_config,
        connection_args=connection_args,
    )
    return MilvusClient(**connection_args)


def _build_ranking_chat_model(*, runtime_config: KnowledgeSearchRuntimeConfig) -> Any:
    """构造知识片段排序使用的普通聊天模型。

    Args:
        runtime_config: 当前请求生效的知识检索运行时配置。

    Returns:
        可执行 ``invoke`` 的普通聊天模型实例。

    Raises:
        ServiceException: 当排序模型名缺失时抛出。
    """

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


def _search_knowledge_hits(
        *,
        question: str,
        final_top_k: int,
        runtime_config: KnowledgeSearchRuntimeConfig,
        ranking_enabled: bool,
) -> list[KnowledgeSearchHit]:
    """执行多知识库聚合向量检索。

    Args:
        question: 当前用于向量检索的问题文本。
        final_top_k: 最终期望返回的命中数量。
        runtime_config: 当前请求生效的知识检索运行时配置。
        ranking_enabled: 当前阶段是否按排序策略扩充候选池。

    Returns:
        聚合去重并按当前阶段策略裁剪后的候选命中列表。
    """

    return retriever_module.search_knowledge_hits(
        question=question,
        final_top_k=final_top_k,
        runtime_config=runtime_config,
        ranking_enabled=ranking_enabled,
        build_milvus_client=_build_rag_milvus_client,
        build_embedding_client=_build_rag_embedding_client,
    )


def _rewrite_question_for_knowledge_search(question: str) -> str:
    """将原始问题改写为更适合向量检索的查询语句。

    Args:
        question: 用户原始问题。

    Returns:
        改写后的检索语句；若改写失败则回退原始问题。
    """

    return rewrite_module.rewrite_question_for_knowledge_search(question)


def _rank_hits_with_chat_model(
        *,
        query: str,
        hits: list[KnowledgeSearchHit],
        runtime_config: KnowledgeSearchRuntimeConfig,
        final_top_k: int,
) -> list[KnowledgeSearchHit]:
    """使用普通聊天模型对候选知识片段进行排序。

    Args:
        query: 排序阶段使用的用户问题。
        hits: 待排序候选片段列表。
        runtime_config: 当前请求生效的知识检索运行时配置。
        final_top_k: 最终期望返回的命中数量。

    Returns:
        排序后的命中列表；当模型返回不足时自动补齐。
    """

    return ranking_module.rank_hits_with_chat_model(
        query=query,
        hits=hits,
        runtime_config=runtime_config,
        final_top_k=final_top_k,
        build_ranking_chat_model=_build_ranking_chat_model,
    )


def _query_knowledge(
        *,
        vector_question: str,
        ranking_question: str,
        top_k: int | None,
) -> list[KnowledgeSearchHit]:
    """统一执行知识库查询与可选排序。

    Args:
        vector_question: 用于向量召回的问题文本。
        ranking_question: 用于排序阶段的问题文本。
        top_k: 调用方显式传入的最终返回条数。

    Returns:
        最终返回给上层调用方的知识片段列表。
    """

    final_top_k = runtime_module.resolve_final_top_k(top_k)
    runtime_config = runtime_module.resolve_runtime_config()

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
    """使用原始问题直接检索知识库。

    Args:
        question: 用户原始问题。
        top_k: 调用方显式传入的最终返回条数；未传时走 Redis 或默认值。

    Returns:
        规范化后的知识片段命中列表。

    Raises:
        ServiceException: 当问题非法或检索失败时抛出。
    """

    normalized_question = normalize_question(question)
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
    """先改写问题，再检索知识库。

    Args:
        question: 用户原始问题。
        top_k: 调用方显式传入的最终返回条数；未传时走 Redis 或默认值。

    Returns:
        规范化后的知识片段命中列表。

    Raises:
        ServiceException: 当问题非法或检索失败时抛出。
    """

    normalized_question = normalize_question(question)
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


def format_knowledge_search_hits(hits: list[KnowledgeSearchHit]) -> str:
    """将结构化命中结果渲染为 Agent 可消费的文本块。

    Args:
        hits: 已排序的知识片段命中列表。

    Returns:
        适合直接拼接到 Agent 上下文中的文本块；未命中时返回固定提示语。
    """

    if not hits:
        return "未检索到相关知识。"

    lines = ["已检索到以下知识片段："]
    consumed_chars = 0
    rendered_count = 0
    for hit in hits:
        remaining_chars = RAG_MAX_CONTEXT_CHARS - consumed_chars
        if remaining_chars <= 0:
            break

        rendered_content = trim_content_to_budget(
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
