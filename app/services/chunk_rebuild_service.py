from __future__ import annotations

from dataclasses import dataclass
from time import time

from pymilvus import exceptions as milvus_exceptions

from app.core.codes import ResponseCode
from app.core.database import get_milvus_client
from app.core.exception.exceptions import ServiceException
from app.core.llms import create_embedding_model
from app.core.mq.state.chunk_rebuild_version_store import get_latest_version as get_chunk_edit_latest_version
from app.repositories import vector_repository
from app.utils.token_utills import TokenUtils

# 单切片编辑重建允许的最大 token 数，超限直接拒绝向量化。
EMBED_MAX_TOKEN_SIZE = 8192
# 读取 Milvus 原始向量行时所需的字段列表。
CHUNK_REBUILD_OUTPUT_FIELDS = [
    "id",
    "document_id",
    "chunk_index",
    "content",
    "char_count",
    "embedding",
    "chunk_strategy",
    "chunk_size",
    "token_size",
    "status",
    "source_hash",
    "created_at_ts",
]


@dataclass(frozen=True)
class ChunkRebuildSuccessResult:
    """单切片重建成功结果。

    Attributes:
        embedding_dim: 本次写入向量的维度。
    """

    embedding_dim: int


class ChunkRebuildMessageStaleError(Exception):
    """表示切片重建任务已被更新版本替代。"""

    def __init__(self, *, vector_id: int, version: int, latest_version: int) -> None:
        """初始化过期任务异常。

        Args:
            vector_id: Milvus 向量主键 ID。
            version: 当前任务版本号。
            latest_version: Redis 中记录的最新版本号。
        """
        self.vector_id = vector_id
        self.version = version
        self.latest_version = latest_version
        super().__init__(
            "切片重建任务已过期，"
            f"vector_id={vector_id}, message_version={version}, latest_version={latest_version}"
        )


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


def _ensure_latest_chunk_edit_version(*, vector_id: int, version: int) -> None:
    """在写入前确认当前任务仍然是该切片的最新版本。

    Args:
        vector_id: Milvus 向量主键 ID。
        version: 当前任务版本号。

    Raises:
        ChunkRebuildMessageStaleError: 当前任务已落后于 Redis 最新版本时抛出。
    """
    latest_version = get_chunk_edit_latest_version(vector_id=vector_id)
    if latest_version is None or version >= latest_version:
        return
    raise ChunkRebuildMessageStaleError(
        vector_id=vector_id,
        version=version,
        latest_version=latest_version,
    )


def rebuild_document_chunk(
        *,
        knowledge_name: str,
        document_id: int,
        vector_id: int,
        version: int,
        content: str,
        embedding_model: str,
) -> ChunkRebuildSuccessResult:
    """按向量主键重建单个文档切片的内容与向量。

    Args:
        knowledge_name: 目标知识库名称。
        document_id: 业务文档 ID。
        vector_id: Milvus 向量主键 ID。
        version: 当前切片编辑版本号。
        content: 新的切片内容。
        embedding_model: 本次任务使用的向量模型名称。

    Returns:
        ChunkRebuildSuccessResult: 重建成功后的摘要结果。

    Raises:
        ServiceException: collection 不存在、向量记录不存在、字段校验失败、
            embedding 失败或 Milvus 写入失败时抛出。
        ChunkRebuildMessageStaleError: 写入前发现当前任务已被更新版本替代时抛出。
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

    try:
        rows = client.get(
            collection_name=knowledge_name,
            ids=vector_id,
            output_fields=CHUNK_REBUILD_OUTPUT_FIELDS,
        )
    except milvus_exceptions.MilvusException as exc:
        raise ServiceException(
            code=ResponseCode.OPERATION_FAILED,
            message=f"读取文档切片失败: {exc}",
        ) from exc

    if not rows:
        raise ServiceException(
            code=ResponseCode.NOT_FOUND,
            message="向量记录不存在",
        )

    persisted_row = dict(rows[0])
    persisted_document_id = int(persisted_row.get("document_id") or 0)
    if persisted_document_id != document_id:
        raise ServiceException(
            code=ResponseCode.BAD_REQUEST,
            message="向量记录与文档ID不匹配",
        )

    embedding = _embed_content(
        content=content,
        embedding_client=embedding_client,
    )

    updated_row = dict(persisted_row)
    updated_row["id"] = int(updated_row.get("id") or vector_id)
    updated_row["document_id"] = persisted_document_id
    updated_row["content"] = content
    updated_row["char_count"] = len(content)
    updated_row["embedding"] = embedding
    updated_row["source_hash"] = None
    updated_row["created_at_ts"] = int(time() * 1000)

    _ensure_latest_chunk_edit_version(vector_id=vector_id, version=version)

    try:
        client.upsert(
            collection_name=knowledge_name,
            data=[updated_row],
        )
    except milvus_exceptions.MilvusException as exc:
        raise ServiceException(
            code=ResponseCode.OPERATION_FAILED,
            message=f"更新文档切片失败: {exc}",
        ) from exc

    return ChunkRebuildSuccessResult(embedding_dim=embedding_dim)
