from pymilvus import (
    CollectionSchema,
    DataType,
    FieldSchema,
    exceptions as milvus_exceptions, MilvusClient,
)

from app.core.codes import ResponseCode
from app.core.exceptions import ServiceException
from app.core.milvus import get_milvus_client
from app.core.llm import create_embedding_client
from app.utils.log import PrintLogger
from app.utils.token_utills import TokenUtils

DEFAULT_CONTENT_MAX_LENGTH = 65535  # 内容字段最大长度
DEFAULT_VECTOR_INDEX_TYPE = "AUTOINDEX"
DEFAULT_VECTOR_METRIC_TYPE = "COSINE"


def _build_index_params(client: "MilvusClient"):
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="document_id", index_type="STL_SORT")
    index_params.add_index(
        field_name="embedding",
        index_type=DEFAULT_VECTOR_INDEX_TYPE,
        metric_type=DEFAULT_VECTOR_METRIC_TYPE,
    )
    return index_params


def _build_collection_schema(embedding_dim: int, description: str) -> CollectionSchema:
    """
    构建知识库向量库的 schema。

    Args:
        embedding_dim: 向量维度
        description: knowledge 描述

    Returns:
        Milvus schema 对象
    """
    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            description="主键（自动生成）",
            is_primary=True,
            auto_id=True,
        ),
        FieldSchema(
            name="document_id",
            dtype=DataType.INT64,
            description="文档ID",
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            description="向量检索字段",
            dim=embedding_dim,
        ),
        FieldSchema(
            name="content",
            dtype=DataType.VARCHAR,
            description="chunk 文本内容",
            max_length=DEFAULT_CONTENT_MAX_LENGTH,
        ),
    ]
    return CollectionSchema(fields=fields, description=description or "")


def create_collection(
        knowledge_name: str, embedding_dim: int, description: str
) -> None:
    """
    创建 Milvus 知识库并应用业务字段 schema。

    Args:
        knowledge_name: knowledge 名称
        embedding_dim: 向量维度
        description: knowledge 描述

    Raises:
        ServiceException: knowledge 已存在或创建失败
    """
    client = get_milvus_client()
    try:
        if client.has_collection(knowledge_name):
            raise ServiceException(
                code=ResponseCode.OPERATION_FAILED, message="knowledge 已存在"
            )
        schema = _build_collection_schema(embedding_dim, description)
        index_params = _build_index_params(client)
        client.create_collection(
            collection_name=knowledge_name,
            schema=schema,
            index_params=index_params,
        )
    except milvus_exceptions.MilvusException as exc:
        raise ServiceException(
            code=ResponseCode.OPERATION_FAILED, message=f"创建 knowledge 失败: {exc}"
        ) from exc


def delete_collection(knowledge_name: str) -> None:
    """
    删除 Milvus 知识库。

    Args:
        knowledge_name: knowledge 名称

    Raises:
        ServiceException: knowledge 不存在或删除失败
    """
    client = get_milvus_client()
    try:
        if not client.has_collection(knowledge_name):
            raise ServiceException(
                code=ResponseCode.NOT_FOUND, message="knowledge 不存在"
            )
        client.drop_collection(knowledge_name)
    except milvus_exceptions.MilvusException as exc:
        raise ServiceException(
            code=ResponseCode.OPERATION_FAILED, message=f"删除 knowledge 失败: {exc}"
        ) from exc


def ensure_collection_exists(knowledge_name: str) -> None:
    """
    校验 Milvus 知识库存在。

    Args:
        knowledge_name: knowledge 名称

    Raises:
        ServiceException: knowledge 不存在或查询失败
    """
    client = get_milvus_client()
    try:
        if not client.has_collection(knowledge_name):
            raise ServiceException(
                code=ResponseCode.NOT_FOUND, message="知识库不存在"
            )
    except milvus_exceptions.MilvusException as exc:
        raise ServiceException(
            code=ResponseCode.OPERATION_FAILED, message=f"查询知识库失败: {exc}"
        ) from exc


def insert_embeddings(
    knowledge_name: str,
    document_id: int,
    embeddings: list[list[float]],
    texts: list[str],
) -> None:
    """
    写入向量数据到 Milvus。

    Args:
        knowledge_name: knowledge 名称
        document_id: 文档ID
        embeddings: 向量列表
        texts: 对应的文本列表

    Raises:
        ServiceException: 写入失败或数据不一致
    """
    if len(embeddings) != len(texts):
        raise ServiceException(
            code=ResponseCode.OPERATION_FAILED,
            message="向量数量与文本数量不一致",
        )
    if not embeddings:
        return

    data = [
        {
            "document_id": document_id,
            "embedding": embedding,
            "content": text,
        }
        for embedding, text in zip(embeddings, texts, strict=True)
    ]
    client = get_milvus_client()
    try:
        client.insert(collection_name=knowledge_name, data=data)
    except milvus_exceptions.MilvusException as exc:
        raise ServiceException(
            code=ResponseCode.OPERATION_FAILED, message=f"写入向量失败: {exc}"
        ) from exc


def embed_texts(texts: list[str]) -> list[list[float]]:
    counts = TokenUtils.count_tokens_list(texts)
    for count in counts:
        if count > 8196:
            # todo 超出长度限制这边应该截断或者直接拒绝
            PrintLogger.error("文本长度超出限制")
            pass

    embeddings_model = create_embedding_client()
    try:
        result = embeddings_model.embed_documents(texts)
    except Exception as exc:
        raise ServiceException(f"嵌入文本失败: {exc}")
    return result
