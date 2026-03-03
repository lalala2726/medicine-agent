from __future__ import annotations

from pymilvus import (
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    exceptions as milvus_exceptions,
)

from app.core.codes import ResponseCode
from app.core.database import get_milvus_client
from app.core.exception.exceptions import ServiceException

DEFAULT_CONTENT_MAX_LENGTH = 65535  # 内容字段最大长度
DEFAULT_CHUNK_STRATEGY_MAX_LENGTH = 32  # 切片策略字段最大长度
DEFAULT_SOURCE_HASH_MAX_LENGTH = 64  # 源文本哈希字段最大长度（sha256）
DEFAULT_VECTOR_INDEX_TYPE = "AUTOINDEX"
DEFAULT_VECTOR_METRIC_TYPE = "COSINE"
COUNT_FALLBACK_LIMIT = 1000


def _build_index_params(client: MilvusClient):
    """
    功能描述:
        构建 Milvus 集合索引参数，包含标量过滤索引与向量检索索引。

    参数说明:
        client (MilvusClient): Milvus 客户端实例。

    返回值:
        Any: Milvus 索引参数对象（由客户端 SDK 提供）。

    异常说明:
        由调用方统一处理底层异常。
    """
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
    功能描述:
        构建知识库向量集合 schema（标准 11 字段版本）。

    参数说明:
        embedding_dim (int): 向量维度。
        description (str): 集合描述文案。

    返回值:
        CollectionSchema: Milvus 集合 schema 对象。

    异常说明:
        无。
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
            name="chunk_no",
            dtype=DataType.INT64,
            description="文档内切片顺序号（从 1 开始）",
            nullable=True,
        ),
        FieldSchema(
            name="content",
            dtype=DataType.VARCHAR,
            description="chunk 文本内容",
            max_length=DEFAULT_CONTENT_MAX_LENGTH,
        ),
        FieldSchema(
            name="char_count",
            dtype=DataType.INT32,
            description="chunk 字符数量统计",
            nullable=True,
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            description="向量检索字段",
            dim=embedding_dim,
        ),
        FieldSchema(
            name="chunk_strategy",
            dtype=DataType.VARCHAR,
            description="切片策略快照",
            max_length=DEFAULT_CHUNK_STRATEGY_MAX_LENGTH,
            nullable=True,
        ),
        FieldSchema(
            name="chunk_size",
            dtype=DataType.INT32,
            description="切片大小参数快照",
            nullable=True,
        ),
        FieldSchema(
            name="token_size",
            dtype=DataType.INT32,
            description="token 切片大小参数快照",
            nullable=True,
        ),
        FieldSchema(
            name="source_hash",
            dtype=DataType.VARCHAR,
            description="源文本哈希（sha256）",
            max_length=DEFAULT_SOURCE_HASH_MAX_LENGTH,
            nullable=True,
        ),
        FieldSchema(
            name="created_at_ts",
            dtype=DataType.INT64,
            description="切片写入时间戳（毫秒）",
            nullable=True,
        ),
    ]
    return CollectionSchema(fields=fields, description=description or "")


def _extract_count(result: list[dict]) -> int | None:
    """
    功能描述:
        从 Milvus 查询结果中提取 count 聚合值。

    参数说明:
        result (list[dict]): Milvus 查询返回的记录列表。

    返回值:
        int | None:
            - 解析成功返回计数值；
            - 无法解析返回 None。

    异常说明:
        无。
    """
    if not result:
        return 0
    first = result[0]
    if isinstance(first, dict):
        if "count(*)" in first:
            return int(first["count(*)"])
        if "count" in first:
            return int(first["count"])
    return None


def _count_by_filter(client: MilvusClient, knowledge_name: str, filter_expr: str) -> int:
    """
    功能描述:
        按过滤条件统计集合记录数，优先使用 count 聚合，失败时回退分页计数。

    参数说明:
        client (MilvusClient): Milvus 客户端实例。
        knowledge_name (str): 集合名称。
        filter_expr (str): 过滤表达式。

    返回值:
        int: 命中记录总数。

    异常说明:
        ServiceException: 底层查询失败时抛出。
    """
    try:
        result = client.query(
            collection_name=knowledge_name,
            filter=filter_expr,
            output_fields=["count(*)"],
        )
        count_value = _extract_count(result)
        if count_value is not None:
            return count_value
    except milvus_exceptions.MilvusException as exc:
        raise ServiceException(
            code=ResponseCode.OPERATION_FAILED,
            message=f"查询知识库失败: {exc}",
        ) from exc

    total = 0
    offset = 0
    while True:
        try:
            batch = client.query(
                collection_name=knowledge_name,
                filter=filter_expr,
                output_fields=["id"],
                limit=COUNT_FALLBACK_LIMIT,
                offset=offset,
            )
        except milvus_exceptions.MilvusException as exc:
            raise ServiceException(
                code=ResponseCode.OPERATION_FAILED,
                message=f"查询知识库失败: {exc}",
            ) from exc
        if not batch:
            break
        batch_len = len(batch)
        total += batch_len
        if batch_len < COUNT_FALLBACK_LIMIT:
            break
        offset += batch_len
    return total


def create_collection(knowledge_name: str, embedding_dim: int, description: str) -> None:
    """
    功能描述:
        创建知识库对应的 Milvus 集合并初始化索引。

    参数说明:
        knowledge_name (str): 集合名称。
        embedding_dim (int): 向量维度。
        description (str): 集合描述。

    返回值:
        None: 创建成功无返回值。

    异常说明:
        ServiceException:
            - 集合已存在时抛出；
            - Milvus 创建失败时抛出。
    """
    client = get_milvus_client()
    try:
        if client.has_collection(knowledge_name):
            raise ServiceException(
                code=ResponseCode.OPERATION_FAILED,
                message="knowledge 已存在",
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
            code=ResponseCode.OPERATION_FAILED,
            message=f"创建 knowledge 失败: {exc}",
        ) from exc


def delete_collection(knowledge_name: str) -> None:
    """
    功能描述:
        删除知识库对应的 Milvus 集合。

    参数说明:
        knowledge_name (str): 集合名称。

    返回值:
        None: 删除成功无返回值。

    异常说明:
        ServiceException:
            - 集合不存在时抛出；
            - Milvus 删除失败时抛出。
    """
    client = get_milvus_client()
    try:
        if not client.has_collection(knowledge_name):
            raise ServiceException(
                code=ResponseCode.NOT_FOUND,
                message="knowledge 不存在",
            )
        client.drop_collection(knowledge_name)
    except milvus_exceptions.MilvusException as exc:
        raise ServiceException(
            code=ResponseCode.OPERATION_FAILED,
            message=f"删除 knowledge 失败: {exc}",
        ) from exc


def ensure_collection_exists(knowledge_name: str) -> None:
    """
    功能描述:
        校验指定知识库集合是否存在。

    参数说明:
        knowledge_name (str): 集合名称。

    返回值:
        None: 集合存在时无返回值。

    异常说明:
        ServiceException:
            - 集合不存在时抛出；
            - Milvus 查询失败时抛出。
    """
    client = get_milvus_client()
    try:
        if not client.has_collection(knowledge_name):
            raise ServiceException(
                code=ResponseCode.NOT_FOUND,
                message="知识库不存在",
            )
    except milvus_exceptions.MilvusException as exc:
        raise ServiceException(
            code=ResponseCode.OPERATION_FAILED,
            message=f"查询知识库失败: {exc}",
        ) from exc


def insert_embeddings(
        knowledge_name: str,
        document_id: int,
        embeddings: list[list[float]],
        texts: list[str],
) -> None:
    """
    功能描述:
        将文档向量批量写入 Milvus。

    参数说明:
        knowledge_name (str): 集合名称。
        document_id (int): 文档 ID。
        embeddings (list[list[float]]): 向量列表。
        texts (list[str]): 文本列表。

    返回值:
        None: 写入成功无返回值。

    异常说明:
        ServiceException:
            - 向量数量与文本数量不一致时抛出；
            - Milvus 写入失败时抛出。
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
            code=ResponseCode.OPERATION_FAILED,
            message=f"写入向量失败: {exc}",
        ) from exc


def list_document_chunks(
        knowledge_name: str,
        document_id: int,
        page_num: int,
        page_size: int,
) -> tuple[list[dict], int]:
    """
    功能描述:
        分页查询指定文档切片记录。

    参数说明:
        knowledge_name (str): 集合名称。
        document_id (int): 文档 ID。
        page_num (int): 页码（从 1 开始）。
        page_size (int): 每页数量。

    返回值:
        tuple[list[dict], int]:
            - 第 1 项: 当前页切片列表；
            - 第 2 项: 总条数。

    异常说明:
        ServiceException:
            - page_num/page_size 非法时抛出；
            - Milvus 查询失败时抛出。
    """
    if page_num <= 0 or page_size <= 0:
        raise ServiceException(
            code=ResponseCode.BAD_REQUEST,
            message="page_num 和 page_size 必须大于 0",
        )

    client = get_milvus_client()
    filter_expr = f"document_id == {document_id}"
    total = _count_by_filter(client, knowledge_name, filter_expr)
    offset = (page_num - 1) * page_size
    try:
        rows = client.query(
            collection_name=knowledge_name,
            filter=filter_expr,
            output_fields=[
                "id",
                "document_id",
                "chunk_no",
                "content",
                "char_count",
                "chunk_strategy",
                "chunk_size",
                "token_size",
                "source_hash",
                "created_at_ts",
            ],
            limit=page_size,
            offset=offset,
        )
    except milvus_exceptions.MilvusException as exc:
        raise ServiceException(
            code=ResponseCode.OPERATION_FAILED,
            message=f"查询知识库失败: {exc}",
        ) from exc
    return rows or [], total


def delete_document_chunks(knowledge_name: str, document_id: int) -> None:
    """
    功能描述:
        删除指定文档在集合中的全部切片记录。

    参数说明:
        knowledge_name (str): 集合名称。
        document_id (int): 文档 ID。

    返回值:
        None: 删除成功无返回值。

    异常说明:
        ServiceException: Milvus 删除失败时抛出。
    """
    client = get_milvus_client()
    filter_expr = f"document_id == {document_id}"
    try:
        client.delete(
            collection_name=knowledge_name,
            filter=filter_expr,
        )
    except milvus_exceptions.MilvusException as exc:
        raise ServiceException(
            code=ResponseCode.OPERATION_FAILED,
            message=f"删除文档切片失败: {exc}",
        ) from exc
