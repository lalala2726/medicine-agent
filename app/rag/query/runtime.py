from __future__ import annotations

from dataclasses import dataclass

from app.core.codes import ResponseCode
from app.core.config_sync import get_current_agent_config_snapshot
from app.core.exception.exceptions import ServiceException
from app.core.llms.common import resolve_llm_value
from app.core.llms.provider import LlmProvider, resolve_provider
from app.rag.query.constants import RAG_DEFAULT_FINAL_TOP_K, RAG_MAX_FINAL_TOP_K, RAG_MAX_KNOWLEDGE_NAMES


@dataclass(frozen=True)
class KnowledgeSearchRuntimeConfig:
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


def normalize_top_k(top_k: int | None) -> int | None:
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


def resolve_final_top_k(explicit_top_k: int | None) -> int:
    """解析当前请求最终生效的返回条数。

    Args:
        explicit_top_k: 调用方显式传入的最终返回条数。

    Returns:
        实际生效的最终返回条数。
    """

    normalized_explicit_top_k = normalize_top_k(explicit_top_k)
    if normalized_explicit_top_k is not None:
        return normalized_explicit_top_k

    configured_top_k = get_current_agent_config_snapshot().get_knowledge_top_k()
    if configured_top_k is not None:
        return configured_top_k
    return RAG_DEFAULT_FINAL_TOP_K


def _resolve_provider_embedding_fallback_model_name(provider_type: str) -> str | None:
    """读取当前 provider 对应的 embedding 环境变量模型名。

    Args:
        provider_type: 当前请求生效的 provider 类型。

    Returns:
        当前 provider 对应的 embedding 模型名称；未配置时返回 ``None``。
    """

    resolved_provider = resolve_provider(provider_type)
    if resolved_provider is LlmProvider.OPENAI:
        return resolve_llm_value(name="OPENAI_EMBEDDING_MODEL")
    if resolved_provider is LlmProvider.ALIYUN:
        return resolve_llm_value(name="DASHSCOPE_EMBEDDING_MODEL")
    return resolve_llm_value(name="VOLCENGINE_LLM_EMBEDDING_MODEL")


def resolve_runtime_config() -> KnowledgeSearchRuntimeConfig:
    """解析当前知识检索链所需的运行时配置。

    Returns:
        当前请求生效的知识检索运行时配置。

    Raises:
        ServiceException: 当缺少必要知识库配置时抛出。
    """

    snapshot = get_current_agent_config_snapshot()
    if not snapshot.is_knowledge_enabled():
        raise ServiceException(
            code=ResponseCode.SERVICE_UNAVAILABLE,
            message="知识库检索未启用",
        )

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

    embedding_model_name = (
            snapshot.get_knowledge_embedding_model_name()
            or _resolve_provider_embedding_fallback_model_name(runtime_config.provider_type)
    )
    if embedding_model_name is None:
        raise ServiceException(
            code=ResponseCode.SERVICE_UNAVAILABLE,
            message="知识库检索未配置 embeddingModel",
        )

    embedding_dim = snapshot.get_knowledge_embedding_dim() or 1024
    return KnowledgeSearchRuntimeConfig(
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
