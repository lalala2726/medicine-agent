import os
import tempfile
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

from pymilvus import CollectionSchema, DataType, FieldSchema, exceptions as milvus_exceptions

from app.core.codes import ResponseCode
from app.core.exceptions import ServiceException
from app.core.milvus import get_milvus_client

# Collection 字段长度配置（按业务字段含义）
DEFAULT_CONTENT_MAX_LENGTH = 65535
DEFAULT_ID_MAX_LENGTH = 256
DEFAULT_SOURCE_MAX_LENGTH = 1024
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
SUPPORTED_IMPORT_EXTENSIONS = {".txt", ".md", ".pdf", ".docx",".html"}


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


def _resolve_filename_from_url(file_url: str) -> str:
    parsed = urlparse(file_url)
    filename = os.path.basename(parsed.path)
    return filename or "downloaded_file"


def _download_file(file_url: str) -> tuple[str, Path]:
    filename = _resolve_filename_from_url(file_url)
    suffix = Path(filename).suffix
    try:
        with urlopen(file_url, timeout=30) as response:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                while True:
                    chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    tmp_file.write(chunk)
                return filename, Path(tmp_file.name)
    except URLError as exc:
        raise ServiceException(f"下载文件失败: {file_url}") from exc


def _validate_import_extension(filename: str) -> None:
    suffix = Path(filename).suffix.lower()
    if not suffix or suffix not in SUPPORTED_IMPORT_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_IMPORT_EXTENSIONS))
        raise ServiceException(f"不支持的文件格式: {suffix or '未知'}，支持: {allowed}")


def _validate_file_not_empty(file_path: Path) -> None:
    if file_path.stat().st_size == 0:
        raise ServiceException("文件为空，无法导入")


def import_knowledge_service(collection_name: str, file_url: list[str]) -> None:
    # 验证 collection 是否存在
    client = get_milvus_client()
    try:
        if not client.has_collection(collection_name):
            raise ServiceException("collection 不存在")
    except milvus_exceptions.MilvusException as exc:
        raise ServiceException(f"查询 collection 失败: {exc}") from exc

    if not file_url:
        raise ServiceException("导入文件不能为空")

    failed_urls: list[str] = []
    for url in file_url:
        try:
            # 访问url下载文件
            filename, file_path = _download_file(url)
            # 验证文件是否可导入文件格式
            _validate_import_extension(filename)
            # 验证文件是否为空
            _validate_file_not_empty(file_path)
        except Exception:
            failed_urls.append(url)
            continue
        print(filename)

    if failed_urls:
        print(f"下载或校验失败的URL: {failed_urls}")
    # 开始切片
    # 导入文件到向量数据库中
    pass
