from __future__ import annotations

from dataclasses import dataclass
from time import time

from pymilvus import exceptions as milvus_exceptions

from app.core.codes import ResponseCode
from app.core.database import get_milvus_client
from app.core.exception.exceptions import ServiceException
from app.core.llms import create_embedding_model
from app.repositories import vector_repository
from app.utils.token_utills import TokenUtils

# 单切片新增允许的最大 token 数，超限直接拒绝向量化。
EMBED_MAX_TOKEN_SIZE = 8192


@dataclass(frozen=True)
class ChunkAddSuccessResult:
    """手工新增切片成功结果。

    Attributes:
        vector_id: 新插入向量的 Milvus 主键 ID。
        chunk_index: 最终切片序号。
        embedding_dim: 本次写入向量的维度。
    """

    vector_id: int
    chunk_index: int
    embedding_dim: int


def _create_embedding_client(*, embedding_model: str, embedding_dim: int):
    """创建向量模型客户端。

    Args:
        embedding_model: 向量模型名称。
        embedding_dim: 目标向量维度。

    Returns:
        Any: 可执行 ``embed_documents`` 的 embedding 客户端。

    Raises:
        ServiceException: 模型初始化失败时抛出。
    """
    try:
        return create_embedding_model(
            model=embedding_model,
            dimensions=embedding_dim,
        )
    except Exception as exc:
        raise ServiceException(message=f"初始化向量模型失败: {exc}") from exc


def _embed_content(*, content: str, embedding_client) -> list[float]:
    """对单条切片内容执行向量化。

    Args:
        content: 待向量化的切片文本。
        embedding_client: embedding 模型客户端。

    Returns:
        list[float]: 生成后的单条向量。

    Raises:
        ServiceException: token 超限、向量化失败或结果为空时抛出。
    """
    token_count = TokenUtils.count_tokens(content)
    if token_count > EMBED_MAX_TOKEN_SIZE:
        raise ServiceException(
            message=(
                "文本超出最大 token 数限制，"
                f"最大 token 数为 {EMBED_MAX_TOKEN_SIZE}, 当前 token 数为 {token_count}"
            ),
        )
    try:
        embeddings = embedding_client.embed_documents([content])
    except Exception as exc:
        raise ServiceException(message=f"嵌入文本失败: {exc}") from exc
    if not embeddings:
        raise ServiceException(message="嵌入结果为空")
    return embeddings[0]


def _get_next_chunk_index(client, knowledge_name: str, document_id: int) -> int:
    """计算该文档下一个可用的 chunk_index。

    通过查询当前文档已有切片的最大 chunk_index + 1 确定。

    Args:
        client: Milvus 客户端实例。
        knowledge_name: 集合名称。
        document_id: 文档 ID。

    Returns:
        int: 下一个可用的 chunk_index（从 1 开始）。
    """
    try:
        rows = client.query(
            collection_name=knowledge_name,
            filter=f"document_id == {document_id}",
            output_fields=["chunk_index"],
            limit=16384,
        )
    except milvus_exceptions.MilvusException as exc:
        raise ServiceException(
            code=ResponseCode.OPERATION_FAILED,
            message=f"查询文档切片失败: {exc}",
        ) from exc

    if not rows:
        return 1

    max_index = max(int(row.get("chunk_index") or 0) for row in rows)
    return max_index + 1


def add_document_chunk(
        *,
        knowledge_name: str,
        document_id: int,
        content: str,
        embedding_model: str,
) -> ChunkAddSuccessResult:
    """向量化并插入一条新的文档切片。

    Args:
        knowledge_name: 目标知识库名称。
        document_id: 业务文档 ID。
        content: 新增切片内容。
        embedding_model: 本次任务使用的向量模型名称。

    Returns:
        ChunkAddSuccessResult: 新增成功后的摘要结果，包含 vector_id 和 chunk_index。

    Raises:
        ServiceException: collection 不存在、embedding 失败或 Milvus 写入失败时抛出。
    """
    vector_repository.ensure_collection_exists(knowledge_name=knowledge_name)
    embedding_dim = vector_repository.get_collection_embedding_dim(
        knowledge_name=knowledge_name,
    )
    embedding_client = _create_embedding_client(
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
    )
    client = get_milvus_client()

    embedding = _embed_content(
        content=content,
        embedding_client=embedding_client,
    )

    chunk_index = _get_next_chunk_index(client, knowledge_name, document_id)

    new_row = {
        "document_id": document_id,
        "chunk_index": chunk_index,
        "content": content,
        "char_count": len(content),
        "embedding": embedding,
        "chunk_strategy": None,
        "chunk_size": None,
        "token_size": None,
        "status": 0,
        "source_hash": None,
        "created_at_ts": int(time() * 1000),
    }

    try:
        insert_result = client.insert(
            collection_name=knowledge_name,
            data=[new_row],
        )
    except milvus_exceptions.MilvusException as exc:
        raise ServiceException(
            code=ResponseCode.OPERATION_FAILED,
            message=f"插入文档切片失败: {exc}",
        ) from exc

    # 从插入结果中提取主键 ID
    vector_id = _extract_primary_key(insert_result)

    return ChunkAddSuccessResult(
        vector_id=vector_id,
        chunk_index=chunk_index,
        embedding_dim=embedding_dim,
    )


def _extract_primary_key(insert_result) -> int:
    """从 Milvus insert 结果中提取主键 ID。

    Args:
        insert_result: Milvus insert 返回的结果对象。

    Returns:
        int: 新插入记录的主键 ID。

    Raises:
        ServiceException: 无法提取主键 ID 时抛出。
    """
    # pymilvus insert 返回 {"insert_count": N, "ids": [id1, ...]}
    if isinstance(insert_result, dict):
        ids = insert_result.get("ids") or insert_result.get("primary_keys") or []
        if ids:
            return int(ids[0])

    # 兼容旧版 pymilvus 返回 MutationResult 对象
    if hasattr(insert_result, "primary_keys"):
        pks = insert_result.primary_keys
        if pks:
            return int(pks[0])

    raise ServiceException(
        code=ResponseCode.OPERATION_FAILED,
        message="无法从 Milvus 插入结果中提取主键 ID",
    )
