import json
import os
import tempfile
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

from pymilvus import (
    CollectionSchema,
    DataType,
    FieldSchema,
    exceptions as milvus_exceptions,
)

from app.core.codes import ResponseCode
from app.core.exceptions import ServiceException
from app.core.milvus import get_milvus_client
from app.services.file_loader.base import cleanup_temp_assets
from app.services.file_loader.factory import FileLoaderFactory

# Collection 字段长度配置（按业务字段含义）
DEFAULT_CONTENT_MAX_LENGTH = 65535  # 内容字段最大长度
DEFAULT_ID_MAX_LENGTH = 256  # ID 字段最大长度
DEFAULT_SOURCE_MAX_LENGTH = 1024  # 来源字段最大长度
DOWNLOAD_CHUNK_SIZE = 1024 * 1024  # 文件下载块大小：1MB
# 支持导入的文件扩展名集合
SUPPORTED_IMPORT_EXTENSIONS = {
    ".txt",
    ".md",
    ".pdf",
    ".docx",
    ".doc",
    ".pptx",
    ".ppt",
    ".xlsx",
    ".xls",
    ".html",
    ".htm",
}


def _build_collection_schema(embedding_dim: int, description: str) -> CollectionSchema:
    """
    构建知识库向量 collection 的 schema。

    Args:
        embedding_dim: 向量维度
        description: collection 描述

    Returns:
        Milvus collection schema 对象
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
            max_length=DEFAULT_ID_MAX_LENGTH,
        ),
    ]
    return CollectionSchema(fields=fields, description=description or "")


def create_collection(
    collection_name: str, embedding_dim: int, description: str
) -> None:
    """
    创建 Milvus collection 并应用业务字段 schema。

    Args:
        collection_name: collection 名称
        embedding_dim: 向量维度
        description: collection 描述

    Raises:
        ServiceException: collection 已存在或创建失败
    """
    client = get_milvus_client()
    try:
        if client.has_collection(collection_name):
            raise ServiceException(
                code=ResponseCode.OPERATION_FAILED, message="collection 已存在"
            )
        schema = _build_collection_schema(embedding_dim, description)
        client.create_collection(collection_name=collection_name, schema=schema)
    except milvus_exceptions.MilvusException as exc:
        raise ServiceException(
            code=ResponseCode.OPERATION_FAILED, message=f"创建 collection 失败: {exc}"
        ) from exc


def delete_collection(collection_name: str) -> None:
    """
    删除 Milvus collection。

    Args:
        collection_name: collection 名称

    Raises:
        ServiceException: collection 不存在或删除失败
    """
    client = get_milvus_client()
    try:
        if not client.has_collection(collection_name):
            raise ServiceException(
                code=ResponseCode.NOT_FOUND, message="collection 不存在"
            )
        client.drop_collection(collection_name)
    except milvus_exceptions.MilvusException as exc:
        raise ServiceException(
            code=ResponseCode.OPERATION_FAILED, message=f"删除 collection 失败: {exc}"
        ) from exc


def _resolve_filename_from_url(file_url: str) -> str:
    """
    从 URL 中解析出文件名。

    Args:
        file_url: 文件 URL

    Returns:
        解析出的文件名，如果无法解析则返回默认名称
    """
    parsed = urlparse(file_url)
    filename = os.path.basename(parsed.path)
    return filename or "downloaded_file"


def _download_file(file_url: str) -> tuple[str, Path]:
    """
    从 URL 下载文件到临时目录，返回文件名和临时文件路径。

    Args:
        file_url: 文件 URL

    Returns:
        (文件名, 临时文件路径) 元组

    Raises:
        ServiceException: 下载失败
    """
    filename = _resolve_filename_from_url(file_url)
    suffix = Path(filename).suffix
    try:
        with urlopen(file_url, timeout=30) as response:
            # 创建临时文件（不自动删除），需在使用后手动清理
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                while True:
                    chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    tmp_file.write(chunk)
                return filename, Path(tmp_file.name)
    except URLError as exc:
        raise ServiceException(
            code=ResponseCode.OPERATION_FAILED, message=f"下载文件失败: {file_url}"
        ) from exc


def _validate_import_extension(filename: str) -> None:
    """
    验证文件扩展名是否在支持的导入格式列表中。

    Args:
        filename: 文件名

    Raises:
        ServiceException: 文件格式不支持
    """
    suffix = Path(filename).suffix.lower()
    if not suffix or suffix not in SUPPORTED_IMPORT_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_IMPORT_EXTENSIONS))
        raise ServiceException(
            code=ResponseCode.BAD_REQUEST,
            message=f"不支持的文件格式: {suffix or '未知'}，支持: {allowed}",
        )


def _validate_file_not_empty(file_path: Path) -> None:
    """
    验证文件不为空。

    Args:
        file_path: 文件路径

    Raises:
        ServiceException: 文件为空
    """
    if file_path.stat().st_size == 0:
        raise ServiceException(
            code=ResponseCode.BAD_REQUEST, message="文件为空，无法导入"
        )


def import_knowledge_service(collection_name: str, file_url: list[str]) -> dict:
    """
    知识库导入服务：批量下载、解析文件并返回结果。

    Args:
        collection_name: Milvus collection 名称
        file_url: 待导入的文件 URL 列表

    Returns:
        包含成功解析结果和失败 URL 列表的字典
    """
    # 验证 collection 是否存在
    client = get_milvus_client()
    try:
        if not client.has_collection(collection_name):
            raise ServiceException(
                code=ResponseCode.NOT_FOUND, message="collection 不存在"
            )
    except milvus_exceptions.MilvusException as exc:
        raise ServiceException(
            code=ResponseCode.OPERATION_FAILED, message=f"查询 collection 失败: {exc}"
        ) from exc

    if not file_url:
        raise ServiceException(
            code=ResponseCode.BAD_REQUEST, message="导入文件不能为空"
        )

    failed_urls: list[str] = []
    results: list[dict] = []
    for url in file_url:
        file_path: Path | None = None
        try:
            # 1. 从 URL 下载文件到临时目录
            filename, file_path = _download_file(url)
            # 2. 验证文件格式是否支持
            _validate_import_extension(filename)
            # 3. 验证文件不为空
            _validate_file_not_empty(file_path)
            # 4. 解析文件内容（包括图片提取）
            parsed = FileLoaderFactory.parse_file_with_images(
                file_path, source_name=filename
            )
            # 输出解析结果（用于调试）
            print(
                json.dumps(
                    {
                        "file_url": url,
                        "filename": filename,
                        "image_dir": parsed["image_dir"],
                        "pages": parsed["pages"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            results.append(
                {
                    "file_url": url,
                    "filename": filename,
                    "image_dir": parsed["image_dir"],
                    "pages": parsed["pages"],
                }
            )
        except ServiceException as exc:
            failed_urls.append(url)
            print(f"解析失败: {url}，原因: {exc}")
            continue
        finally:
            # 清理下载的临时文件（无论解析成功与否）
            if file_path and file_path.exists():
                file_path.unlink(missing_ok=True)

    if failed_urls:
        print(f"下载或校验失败的URL: {failed_urls}")
    return {
        "results": results,
        "failed_urls": failed_urls,
    }


def cleanup_import_assets(filename: str) -> dict:
    """
    按文件名清理解析产生的临时文件与图片目录。

    Args:
        filename: 文件名

    Returns:
        清理结果统计字典
    """
    return cleanup_temp_assets(filename)
