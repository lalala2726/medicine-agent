import os
import tempfile
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

from app.core.codes import ResponseCode
from app.core.exceptions import ServiceException
from app.core.chunking import ChunkStrategyType, SplitConfig, split_file
from app.core.file_loader.base import cleanup_temp_assets
from app.core.file_loader.factory import FileLoaderFactory
from app.services import vector_service
from app.utils.log import logger

DOWNLOAD_CHUNK_SIZE = 1024 * 1024  # 文件下载块大小：1MB
DEFAULT_CHUNK_SIZE = 500  # 默认切片长度（字符）
DEFAULT_TOKEN_SIZE = 100  # 默认 token 切片长度
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


def create_collection(
        knowledge_name: str, embedding_dim: int, description: str
) -> None:
    """
    创建 Milvus 知识库并应用业务字段 schema。

    Args:
        knowledge_name: knowledge 名称
        embedding_dim: 向量维度
        description: knowledge 描述
    """
    vector_service.create_collection(knowledge_name, embedding_dim, description)


def delete_collection(knowledge_name: str) -> None:
    """
    删除 Milvus 知识库。

    Args:
        knowledge_name: knowledge 名称
    """
    vector_service.delete_collection(knowledge_name)


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


def import_knowledge_service(
        knowledge_name: str,
        document_id: int,
        file_url: list[str],
        chunk_strategy: ChunkStrategyType = ChunkStrategyType.LENGTH,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        token_size: int = DEFAULT_TOKEN_SIZE,
) -> dict:
    """
    知识库导入服务：批量下载、解析文件并返回结果。

    Args:
        knowledge_name: Milvus knowledge 名称
        file_url: 待导入的文件 URL 列表

    Returns:
        包含成功解析结果和失败 URL 列表的字典
        :param knowledge_name:
        :param document_id:
        :param file_url:
        :param chunk_strategy:
        :param chunk_size:
        :param token_size:
    """
    logger.info(
        "开始导入知识库：knowledge_name=%s, document_id=%s, file_count=%s, chunk_strategy=%s, chunk_size=%s, token_size=%s",
        knowledge_name,
        document_id,
        len(file_url) if file_url else 0,
        chunk_strategy.value,
        chunk_size,
        token_size,
    )
    # 验证 knowledge 是否存在
    vector_service.ensure_collection_exists(knowledge_name)

    if not file_url:
        raise ServiceException(
            code=ResponseCode.BAD_REQUEST, message="导入文件不能为空"
        )

    # 下载失败的文件 URL 列表
    failed_urls: list[str] = []
    # 下载结果列表
    results: list[dict] = []
    # 向量写入已暂时关闭，后续需要时再恢复维度校验

    for url in file_url:
        filename: str | None = None
        file_path: Path | None = None
        keep_file = False
        try:
            logger.info("开始处理文件：file_url=%s", url)
            # 1. 从 URL 下载文件到临时目录
            filename, file_path = _download_file(url)
            logger.info(
                "下载完成：file_url=%s, filename=%s, temp_path=%s",
                url,
                filename,
                file_path,
            )
            # 2. 验证文件格式是否支持
            _validate_import_extension(filename)
            # 3. 验证文件不为空
            _validate_file_not_empty(file_path)
            # 4. 解析文件内容（包括图片提取）
            parsed = FileLoaderFactory.parse_file_with_images(
                file_path, source_name=filename
            )
            logger.info(
                "解析完成：filename=%s, pages=%s, image_dir=%s",
                filename,
                len(parsed.get("pages") or []),
                parsed.get("image_dir"),
            )
            # 5. 按策略切片
            effective_chunk_size = (
                token_size
                if chunk_strategy == ChunkStrategyType.TOKEN
                else chunk_size
            )
            chunks = split_file(
                file_path,
                chunk_strategy,
                SplitConfig(chunk_size=effective_chunk_size, chunk_overlap=0),
            )
            logger.info(
                "切片完成：filename=%s, chunk_strategy=%s, chunk_size=%s, chunks=%s",
                filename,
                chunk_strategy.value,
                effective_chunk_size,
                len(chunks),
            )
            # 6. 向量化并写入 Milvus
            texts = [chunk.text for chunk in chunks if chunk.text.strip()]
            if texts:
                logger.info(
                    "已获取文本切片：filename=%s, text_count=%s",
                    filename,
                    len(texts),
                )
            else:
                logger.warning("未生成有效文本，跳过向量写入：filename=%s", filename)
            keep_file = True
            results.append(
                {
                    "file_url": url,
                    "filename": filename,
                    "image_dir": parsed["image_dir"],
                    "pages": parsed["pages"],
                }
            )
            logger.info("文件处理完成：filename=%s, file_url=%s", filename, url)
        except ServiceException as exc:
            logger.error(
                "文件处理失败：file_url=%s, filename=%s, error=%s",
                url,
                filename,
                exc,
            )
            failed_urls.append(url)
            if filename:
                cleanup_temp_assets(filename)
            continue
        finally:
            # 解析失败时清理下载的临时文件，成功则等待统一清理
            if not keep_file and file_path and file_path.exists():
                file_path.unlink(missing_ok=True)
                logger.info("临时文件已清理：temp_path=%s", file_path)
    summary = {
        "results": results,
        "failed_urls": failed_urls,
    }
    return summary


def cleanup_import_assets(filename: str) -> dict:
    """
    按文件名清理解析产生的临时文件与图片目录。

    Args:
        filename: 文件名

    Returns:
        清理结果统计字典
    """
    return cleanup_temp_assets(filename)
