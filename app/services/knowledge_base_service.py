from pymilvus import CollectionSchema, DataType, FieldSchema, exceptions as milvus_exceptions

from app.core.codes import ResponseCode
from app.core.exceptions import ServiceException
from app.core.milvus import get_milvus_client

DEFAULT_CONTENT_MAX_LENGTH = 65535
DEFAULT_ID_MAX_LENGTH = 256
DEFAULT_SOURCE_MAX_LENGTH = 1024


def _build_collection_schema(embedding_dim: int, description: str) -> CollectionSchema:
    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True,
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=embedding_dim,
        ),
        FieldSchema(
            name="content",
            dtype=DataType.VARCHAR,
            max_length=DEFAULT_CONTENT_MAX_LENGTH,
        ),
        FieldSchema(
            name="knowledge_base_id",
            dtype=DataType.VARCHAR,
            max_length=DEFAULT_ID_MAX_LENGTH,
        ),
        FieldSchema(
            name="document_id",
            dtype=DataType.VARCHAR,
            max_length=DEFAULT_ID_MAX_LENGTH,
        ),
        FieldSchema(
            name="chunk_index",
            dtype=DataType.INT64,
        ),
        FieldSchema(
            name="source",
            dtype=DataType.VARCHAR,
            max_length=DEFAULT_SOURCE_MAX_LENGTH,
            is_nullable=True,
        ),
        FieldSchema(
            name="metadata",
            dtype=DataType.JSON,
            is_nullable=True,
        ),
    ]
    return CollectionSchema(fields=fields, description=description or "")


def create_collection(collection_name: str, embedding_dim: int, description: str) -> None:
    client = get_milvus_client()
    try:
        if client.has_collection(collection_name):
            raise ServiceException(code=ResponseCode.OPERATION_FAILED, message="collection 已存在")
        schema = _build_collection_schema(embedding_dim, description)
        client.create_collection(collection_name=collection_name, schema=schema)
    except milvus_exceptions.MilvusException as exc:
        raise ServiceException(code=ResponseCode.OPERATION_FAILED, message=f"创建 collection 失败: {exc}") from exc


def delete_collection(collection_name: str) -> None:
    client = get_milvus_client()
    try:
        if not client.has_collection(collection_name):
            raise ServiceException(code=ResponseCode.NOT_FOUND, message="collection 不存在")
        client.drop_collection(collection_name)
    except milvus_exceptions.MilvusException as exc:
        raise ServiceException(code=ResponseCode.OPERATION_FAILED, message=f"删除 collection 失败: {exc}") from exc
