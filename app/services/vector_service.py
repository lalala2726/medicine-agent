from pymilvus import (
    CollectionSchema,
    DataType,
    FieldSchema,
    exceptions as milvus_exceptions, MilvusClient,
)

from app.core.codes import ResponseCode
from app.core.exceptions import ServiceException
from app.core.llm import create_embedding_client
from app.core.milvus import get_milvus_client
from app.utils.token_utills import TokenUtils

DEFAULT_CONTENT_MAX_LENGTH = 65535  # 内容字段最大长度
DEFAULT_VECTOR_INDEX_TYPE = "AUTOINDEX"
DEFAULT_VECTOR_METRIC_TYPE = "COSINE"
EMBED_BATCH_SIZE = 10  # 向量模型单次最大处理文本数
EMBED_MAX_WORKERS = 5  # 最大并发线程数
EMBED_MAX_TOKEN_SIZE = 8192
COUNT_FALLBACK_LIMIT = 1000


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


def _extract_count(result: list[dict]) -> int | None:
    if not result:
        return 0
    first = result[0]
    if isinstance(first, dict):
        if "count(*)" in first:
            return int(first["count(*)"])
        if "count" in first:
            return int(first["count"])
    return None


def _count_by_filter(
        client: "MilvusClient",
        knowledge_name: str,
        filter_expr: str,
) -> int:
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
            code=ResponseCode.OPERATION_FAILED, message=f"查询知识库失败: {exc}"
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
                code=ResponseCode.OPERATION_FAILED, message=f"查询知识库失败: {exc}"
            ) from exc
        if not batch:
            break
        batch_len = len(batch)
        total += batch_len
        if batch_len < COUNT_FALLBACK_LIMIT:
            break
        offset += batch_len
    return total


def list_document_chunks(
        knowledge_name: str,
        document_id: int,
        page_num: int,
        page_size: int,
) -> tuple[list[dict], int]:
    """
    分页查询指定 document_id 的向量内容（仅返回元信息与文本）。
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
            output_fields=["id", "document_id", "content"],
            limit=page_size,
            offset=offset,
        )
    except milvus_exceptions.MilvusException as exc:
        raise ServiceException(
            code=ResponseCode.OPERATION_FAILED, message=f"查询知识库失败: {exc}"
        ) from exc
    return rows or [], total


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
     对文本列表进行向量化处理。

     说明（以阿里云百炼向量化服务为例）：
     单次向量化请求（一个 batch）支持的最大输入规模为 8192 个 token，
        即同一个请求中所有文本的 token 总数之和不能超过 8192。
     批量处理时，单个 batch 最多支持 10 条文本（batch size ≤ 10），
        该限制与 token 上限同时生效，而不是相乘关系。
     并发调用能力受账号配额与平台限流策略约束，具体并发限制请以
        阿里云百炼控制台展示为准：
        https://bailian.console.aliyun.com/

     在本函数中为了确保不超过 8192 token 或 10 条文本的限制，我们对文本进行切分处理
     但是为了兼顾性能这边使用多线程进行处理

     这边返回的文本的向量并且这边会确保是按照顺序返回

     """
    counts = TokenUtils.count_tokens_list(texts)
    for count in counts:
        if count > EMBED_MAX_TOKEN_SIZE:
            raise ServiceException(
                f"文本超出最大 token 数限制，最大 token 数为 {EMBED_MAX_TOKEN_SIZE}, 当前 token 数为 {count}")

    # 获取向量模型
    embeddings_model = create_embedding_client()
    if not texts:
        return []

    def _split_batches(items: list[str], batch_size: int) -> list[list[str]]:
        """
        将文本列表按指定批次大小进行分割。

        Args:
            items: 待分割的文本列表
            batch_size: 每个批次的大小

        Returns:
            分割后的文本批次列表
        """
        result = []
        for i in range(0, len(items), batch_size):
            result.append(items[i:i + batch_size])
        return result

    def _embed_batch(batch: list[str]) -> list[list[float]]:
        """
        对单个批次的文本进行向量嵌入。

        Args:
            batch: 文本批次列表

        Returns:
            嵌入向量列表
        """
        return embeddings_model.embed_documents(batch)

    batches = _split_batches(texts, EMBED_BATCH_SIZE)
    # 计算文本；列表是否达到需要开启并行处理
    if len(batches) == 1:
        try:
            return _embed_batch(batches[0])
        except Exception as exc:
            raise ServiceException(f"嵌入文本失败: {exc}") from exc

    from concurrent.futures import ThreadPoolExecutor

    results: list[list[float]] = []
    try:
        with ThreadPoolExecutor(max_workers=EMBED_MAX_WORKERS) as executor:
            futures = []
            for batch in batches:
                future = executor.submit(_embed_batch, batch)
                futures.append(future)
            for future in futures:
                results.extend(future.result())
    except Exception as exc:
        raise ServiceException(f"嵌入文本失败: {exc}") from exc
    return results
