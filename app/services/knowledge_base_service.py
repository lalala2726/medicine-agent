from pathlib import Path

from loguru import logger

from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException
from app.rag.chunking import ChunkStrategyType, SplitChunk, SplitConfig, split_text
from app.rag.file_loader import parse_downloaded_file, validate_url_extension
from app.rag.file_loader.types import ParsedPage
from app.services import vector_service
from app.utils.file_utils import FileUtils

DEFAULT_CHUNK_SIZE = 500  # 默认切片长度（字符）
DEFAULT_TOKEN_SIZE = 100  # 默认 token 切片长度
DEFAULT_CHUNK_OVERLAP = 50  # 默认切片重叠长度（字符）


def _download_file(url: str) -> tuple[str, Path]:
    """
    下载文件并返回（文件名，固定下载目录下的文件路径）。

    单独封装该函数是为了测试可替换（monkeypatch），
    同时避免导入流程与底层下载实现强耦合。
    """

    return FileUtils.download_file(url)


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


def delete_knowledge(knowledge_name: str) -> None:
    """
    删除 Milvus 知识库。

    Args:
        knowledge_name: knowledge 名称
    """
    vector_service.delete_collection(knowledge_name)


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
        file_url: list[str] | str,
        chunk_strategy: ChunkStrategyType = ChunkStrategyType.CHARACTER,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        token_size: int = DEFAULT_TOKEN_SIZE,
) -> dict:
    """
    功能描述:
        批量导入知识库文件，执行下载、文件校验、文本解析与切片打印流程。
        当前为调试阶段：仅验证“下载 -> 切片”链路，暂不写入向量数据库。

    参数说明:
        knowledge_name (str): 知识库名称。
        document_id (int): 文档 ID。
        file_url (list[str] | str): 待导入文件 URL 列表或单个 URL 字符串。
        chunk_strategy (ChunkStrategyType): 切片策略类型，默认值为 ChunkStrategyType.CHARACTER。
        chunk_size (int): 字符类切片大小，默认值为 DEFAULT_CHUNK_SIZE。
        token_size (int): token 切片大小，默认值为 DEFAULT_TOKEN_SIZE。

    返回值:
        dict: 导入汇总结果，包含 `results`（成功列表）与 `failed_urls`（失败 URL 列表）。

    异常说明:
        ServiceException: 参数非法、下载失败、文件解析失败或切片失败时抛出。
    """
    normalized_urls = _normalize_import_urls(file_url)
    logger.info(
        "开始导入知识库：knowledge_name={}, document_id={}, file_count={}, chunk_strategy={}, chunk_size={}, token_size={}",
        knowledge_name,
        document_id,
        len(normalized_urls),
        chunk_strategy.value,
        chunk_size,
        token_size,
    )

    if not normalized_urls:
        raise ServiceException(
            code=ResponseCode.BAD_REQUEST, message="导入文件不能为空"
        )

    failed_urls: list[str] = []
    results: list[dict] = []

    for url in normalized_urls:
        filename: str | None = None
        file_path: Path | None = None
        try:
            logger.info("开始处理文件：file_url={}", url)
            source_extension = validate_url_extension(url)
            filename, file_path = _download_file(url)
            logger.info(
                "下载完成：file_url={}, filename={}, file_path={}",
                url,
                filename,
                file_path,
            )
            _validate_file_not_empty(file_path)
            parsed_document = parse_downloaded_file(
                file_path=file_path,
                source_url=url,
            )
            pages = parsed_document.pages
            logger.info(
                "解析完成：filename={}, file_kind={}, mime_type={}, pages={}",
                filename,
                parsed_document.file_kind.value,
                parsed_document.mime_type,
                len(pages),
            )
            effective_chunk_size = (
                token_size
                if chunk_strategy == ChunkStrategyType.TOKEN
                else chunk_size
            )
            chunks = _split_parsed_pages(
                pages,
                chunk_strategy,
                effective_chunk_size,
            )
            logger.info(
                "切片完成：filename={}, chunk_strategy={}, chunk_size={}, chunks={}",
                filename,
                chunk_strategy.value,
                effective_chunk_size,
                len(chunks),
            )
            _print_chunks_to_console(
                filename=filename or "unknown",
                chunks=chunks,
            )

            results.append(
                {
                    "file_url": url,
                    "filename": filename,
                    "source_extension": source_extension,
                    "file_kind": parsed_document.file_kind.value,
                    "mime_type": parsed_document.mime_type,
                    "pages": [page.to_dict() for page in pages],
                    "chunk_count": len(chunks),
                }
            )
        except ServiceException as exc:
            logger.error(
                "文件处理失败：file_url={}, filename={}, error={}",
                url,
                filename,
                exc,
            )
            failed_urls.append(url)
            continue
    return {"results": results, "failed_urls": failed_urls}


def _normalize_import_urls(file_url: list[str] | str) -> list[str]:
    """
    功能描述:
        归一化导入 URL 参数，统一转换为 URL 列表，兼容单个字符串与字符串列表输入。

    参数说明:
        file_url (list[str] | str): 原始 URL 入参。

    返回值:
        list[str]: 清洗后的 URL 列表（去除空白项）。

    异常说明:
        ServiceException: 当参数类型非法或无有效 URL 时抛出。
    """
    if isinstance(file_url, str):
        normalized = [file_url.strip()] if file_url.strip() else []
    elif isinstance(file_url, list):
        normalized = [str(item).strip() for item in file_url if str(item).strip()]
    else:
        raise ServiceException(
            code=ResponseCode.BAD_REQUEST,
            message="file_url 参数类型错误，必须是字符串或字符串列表",
        )
    return normalized


def _print_chunks_to_console(*, filename: str, chunks: list[SplitChunk]) -> None:
    """
    功能描述:
        将切片结果按可读格式打印到控制台，供导入链路调试验证使用。

    参数说明:
        filename (str): 当前文件名，用于控制台分组标识。
        chunks (list[SplitChunk]): 切片结果列表。

    返回值:
        None: 打印完成无返回值。

    异常说明:
        无。打印异常会由 Python 标准输出层自行处理。
    """
    print(f"[chunk-debug] filename={filename}, chunk_count={len(chunks)}")
    for chunk in chunks:
        chunk_text = chunk.text.strip()
        print(
            "[chunk-debug] "
            f"chunk_index={chunk.chunk_index}, "
            f"page_number={chunk.page_number}, "
            f"page_label={chunk.page_label}, "
            f"text={chunk_text}"
        )


def _split_parsed_pages(
        pages: list[ParsedPage],
        chunk_strategy: ChunkStrategyType,
        chunk_size: int,
) -> list[SplitChunk]:
    """
    功能描述:
        遍历解析后的页面列表，对每页文本执行切片，并将页级元数据回填到切片结果中。

    参数说明:
        pages (list[ParsedPage]): 解析后的页面列表。
        chunk_strategy (ChunkStrategyType): 切片策略类型。
        chunk_size (int): 生效切片大小（字符策略时为 chunk_size，token 策略时为 token_size）。

    返回值:
        list[SplitChunk]: 聚合后的切片结果列表。

    异常说明:
        ServiceException: 当切片策略不支持或底层切片依赖缺失时由下游抛出。
    """
    config = SplitConfig(
        chunk_size=chunk_size,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )
    chunks: list[SplitChunk] = []
    for page in pages:
        text = page.text or ""
        if not text.strip():
            continue
        page_chunks = split_text(text, chunk_strategy, config)
        page_metadata = _build_page_chunk_metadata(page)
        for chunk in page_chunks:
            chunk.page_number = page_metadata["page_number"]
            chunk.page_label = page_metadata["page_label"]
            chunk.metadata = {**page_metadata, **chunk.metadata}
        chunks.extend(page_chunks)
    return chunks


def _build_page_chunk_metadata(page: ParsedPage) -> dict:
    """
    功能描述:
        从解析页对象构造标准页级元数据，供切片结果统一写入 metadata。

    参数说明:
        page (ParsedPage): 解析页对象，支持 page_number 与 page_label 字段。

    返回值:
        dict: 标准化页级元数据，包含 page_number、page_label。

    异常说明:
        无。字段缺失或类型异常时使用兜底默认值。
    """
    raw_page_number = page.page_number
    try:
        resolved_page_number = int(raw_page_number)
    except (TypeError, ValueError):
        resolved_page_number = 1
    if resolved_page_number <= 0:
        resolved_page_number = 1

    raw_page_label = page.page_label
    resolved_page_label = raw_page_label if isinstance(raw_page_label, str) else None

    return {
        "page_number": resolved_page_number,
        "page_label": resolved_page_label,
    }


def delete_document(knowledge_name: str, document_id: int) -> None:
    vector_service.ensure_collection_exists(knowledge_name)
    vector_service.delete_document(
        knowledge_name=knowledge_name,
        document_id=document_id,
    )


def list_knowledge_chunks(
        knowledge_name: str,
        document_id: int,
        page_num: int,
        page_size: int,
) -> tuple[list[dict], int]:
    """
    分页查询知识库中的文档切片数据。

    Args:
        knowledge_name: 知识库名
        document_id: 文档ID
        page_num: 页码（从 1 开始）
        page_size: 每页数量

    Returns:
        (rows, total) 元组
    """
    if page_num <= 0 or page_size <= 0:
        raise ServiceException(
            code=ResponseCode.BAD_REQUEST,
            message="page_num 和 page_size 必须大于 0",
        )
    vector_service.ensure_collection_exists(knowledge_name)
    return vector_service.list_document_chunks(
        knowledge_name=knowledge_name,
        document_id=document_id,
        page_num=page_num,
        page_size=page_size,
    )
