from pymilvus import CollectionSchema, DataType, FieldSchema, exceptions as milvus_exceptions

from app.core.codes import ResponseCode
from app.core.exceptions import ServiceException
from app.core.milvus import get_milvus_client

# Collection 字段长度配置（按业务字段含义）
DEFAULT_CONTENT_MAX_LENGTH = 65535
DEFAULT_ID_MAX_LENGTH = 256
DEFAULT_SOURCE_MAX_LENGTH = 1024


def _build_collection_schema(embedding_dim: int, description: str) -> CollectionSchema:
    """构建知识库向量 collection 的 schema。"""
    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            description="主键（自动生成）",
            is_primary=True,
            auto_id=True,
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
        FieldSchema(
            name="knowledge_base_id",
            dtype=DataType.VARCHAR,
            description="知识库隔离标识",
            max_length=DEFAULT_ID_MAX_LENGTH
        )
    ]
    return CollectionSchema(fields=fields, description=description or "")


def create_collection(collection_name: str, embedding_dim: int, description: str) -> None:
    """创建 Milvus collection 并应用业务字段 schema。"""
    client = get_milvus_client()
    try:
        if client.has_collection(collection_name):
            raise ServiceException(code=ResponseCode.OPERATION_FAILED, message="collection 已存在")
        schema = _build_collection_schema(embedding_dim, description)
        client.create_collection(collection_name=collection_name, schema=schema)
    except milvus_exceptions.MilvusException as exc:
        raise ServiceException(code=ResponseCode.OPERATION_FAILED, message=f"创建 collection 失败: {exc}") from exc


def delete_collection(collection_name: str) -> None:
    """删除 Milvus collection。"""
    client = get_milvus_client()
    try:
        if not client.has_collection(collection_name):
            raise ServiceException(code=ResponseCode.NOT_FOUND, message="collection 不存在")
        client.drop_collection(collection_name)
    except milvus_exceptions.MilvusException as exc:
        raise ServiceException(code=ResponseCode.OPERATION_FAILED, message=f"删除 collection 失败: {exc}") from exc
