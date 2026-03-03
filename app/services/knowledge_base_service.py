from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from loguru import logger

from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException
from app.core.llms import create_embedding_model
from app.core.mq.models import KnowledgeImportMessage
from app.core.mq.publisher import publish_import_messages
from app.repositories import vector_repository
from app.rag.chunking import ChunkStrategyType, SplitChunk, SplitConfig, split_text
from app.rag.file_loader import parse_downloaded_file, validate_url_extension
from app.utils.file_utils import FileUtils
from app.utils.token_utills import TokenUtils

DEFAULT_CHUNK_SIZE = 500  # 默认切片长度（字符）
DEFAULT_TOKEN_SIZE = 100  # 默认 token 切片长度
DEFAULT_CHUNK_OVERLAP = 50  # 默认切片重叠长度（字符）
EMBED_BATCH_SIZE = 10  # 向量模型单次最大处理文本数
EMBED_MAX_WORKERS = 5  # 最大并发线程数
EMBED_MAX_TOKEN_SIZE = 8192  # 向量化前单文本最大 token 限制


async def submit_import_to_queue(
        knowledge_name: str,
        document_id: int,
        file_url: list[str] | str,
        chunk_strategy: ChunkStrategyType = ChunkStrategyType.CHARACTER,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        token_size: int = DEFAULT_TOKEN_SIZE,
) -> dict:
    """
    提交导入请求到 RabbitMQ。

    Args:
        knowledge_name: 知识库名称。
        document_id: 文档 ID。
        file_url: 文件 URL（字符串或字符串列表）。
        chunk_strategy: 切片策略，默认 `character`。
        chunk_size: 字符切片大小，默认 `500`。
        token_size: token 切片大小，默认 `100`。

    Returns:
        投递结果字典，包含 `accepted_count` 和 `task_uuids`。

    Raises:
        ServiceException: URL 为空/非法、后缀不支持或 MQ 投递失败。
    """
    normalized_urls = _normalize_import_urls(file_url)
    if not normalized_urls:
        raise ServiceException(
            code=ResponseCode.BAD_REQUEST,
            message="导入文件不能为空",
        )

    messages: list[KnowledgeImportMessage] = []
    for url in normalized_urls:
        validate_url_extension(url)
        messages.append(
            KnowledgeImportMessage.build(
                knowledge_name=knowledge_name,
                document_id=document_id,
                file_url=url,
                chunk_strategy=chunk_strategy,
                chunk_size=chunk_size,
                token_size=token_size,
            )
        )

    await publish_import_messages(messages)
    task_uuids = [message.task_uuid for message in messages]
    logger.info(
        "导入任务投递 MQ 成功：knowledge_name={}, document_id={}, url_count={}, task_uuids={}",
        knowledge_name,
        document_id,
        len(messages),
        task_uuids,
    )
    return {
        "accepted_count": len(messages),
        "task_uuids": task_uuids,
    }


def _split_embed_batches(items: list[str], batch_size: int) -> list[list[str]]:
    """
    将文本按批次切分。

    Args:
        items: 原始文本列表。
        batch_size: 单批最大条数。

    Returns:
        二维列表，每个子列表为一个批次。

    Raises:
        无。
    """
    result: list[list[str]] = []
    for index in range(0, len(items), batch_size):
        result.append(items[index:index + batch_size])
    return result


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    批量文本向量化（含 token 限制与并发）。

    Args:
        texts: 待向量化文本列表。

    Returns:
        向量列表，顺序与 `texts` 一致。

    Raises:
        ServiceException: 文本超过 token 限制或模型调用失败。
    """
    if not texts:
        return []

    token_counts = TokenUtils.count_tokens_list(texts)
    for token_count in token_counts:
        if token_count > EMBED_MAX_TOKEN_SIZE:
            raise ServiceException(
                message=(
                    "文本超出最大 token 数限制，"
                    f"最大 token 数为 {EMBED_MAX_TOKEN_SIZE}, 当前 token 数为 {token_count}"
                ),
            )

    embedding_model = create_embedding_model()

    def _embed_batch(batch: list[str]) -> list[list[float]]:
        """
        执行单批文本向量化。

        Args:
            batch: 单批文本列表。

        Returns:
            当前批次向量结果。

        Raises:
            异常由外层统一转换为 `ServiceException`。
        """
        return embedding_model.embed_documents(batch)

    batches = _split_embed_batches(texts, EMBED_BATCH_SIZE)
    if len(batches) == 1:
        try:
            return _embed_batch(batches[0])
        except Exception as exc:  # pragma: no cover - 依赖外部模型 SDK
            raise ServiceException(message=f"嵌入文本失败: {exc}") from exc

    results: list[list[float]] = []
    try:
        with ThreadPoolExecutor(max_workers=EMBED_MAX_WORKERS) as executor:
            futures = [executor.submit(_embed_batch, batch) for batch in batches]
            for future in futures:
                results.extend(future.result())
    except Exception as exc:  # pragma: no cover - 依赖外部模型 SDK
        raise ServiceException(message=f"嵌入文本失败: {exc}") from exc
    return results


def _download_file(url: str) -> tuple[str, Path]:
    """
    下载文件并返回文件名与本地路径。

    Args:
        url: 远程文件 URL。

    Returns:
        `(filename, file_path)`。

    Raises:
        ServiceException: 下载失败或下载目录配置异常。
    """

    return FileUtils.download_file(url)


def create_collection(
        knowledge_name: str,
        embedding_dim: int,
        description: str,
) -> None:
    """
    创建知识库对应的 Milvus collection。

    Args:
        knowledge_name: 知识库名称。
        embedding_dim: 向量维度。
        description: 知识库描述。

    Returns:
        None。

    Raises:
        ServiceException: collection 已存在或创建失败。
    """
    vector_repository.create_collection(
        knowledge_name=knowledge_name,
        embedding_dim=embedding_dim,
        description=description,
    )


def delete_knowledge(knowledge_name: str) -> None:
    """
    删除知识库对应的 Milvus collection。

    Args:
        knowledge_name: 知识库名称。

    Returns:
        None。

    Raises:
        ServiceException: collection 不存在或删除失败。
    """
    vector_repository.delete_collection(knowledge_name=knowledge_name)


def _validate_file_not_empty(file_path: Path) -> None:
    """
    校验文件非空。

    Args:
        file_path: 本地文件路径。

    Returns:
        None。

    Raises:
        ServiceException: 文件大小为 0。
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
    执行导入主流程：下载 -> 解析 -> 切片 -> 控制台打印。

    Args:
        knowledge_name: 知识库名称。
        document_id: 文档 ID。
        file_url: 文件 URL（字符串或字符串列表）。
        chunk_strategy: 切片策略。
        chunk_size: 字符切片大小。
        token_size: token 切片大小。

    Returns:
        导入结果字典，包含 `results` 和 `failed_urls`。

    Raises:
        ServiceException: 参数非法时抛出（其他文件级异常会记录并计入失败列表）。
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
            parsed_text = parsed_document.text or ""
            logger.info(
                "解析完成：filename={}, file_kind={}, mime_type={}, text_length={}",
                filename,
                parsed_document.file_kind.value,
                parsed_document.mime_type,
                len(parsed_text),
            )
            effective_chunk_size = (
                token_size
                if chunk_strategy == ChunkStrategyType.TOKEN
                else chunk_size
            )
            chunks = _split_parsed_text(
                parsed_text,
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
                    "text_length": len(parsed_text),
                    "chunk_count": len(chunks),
                    "chunks": [chunk.to_dict() for chunk in chunks],
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
        except Exception as exc:
            logger.exception(
                "文件处理异常：file_url={}, filename={}, error={}",
                url,
                filename,
                exc,
            )
            failed_urls.append(url)
            continue
    return {"results": results, "failed_urls": failed_urls}


def _normalize_import_urls(file_url: list[str] | str) -> list[str]:
    """
    标准化导入 URL 入参。

    Args:
        file_url: 原始 URL 入参。

    Returns:
        清洗后的 URL 列表（去除空白项）。

    Raises:
        ServiceException: 参数类型不是字符串或字符串列表。
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
    打印切片调试信息。

    Args:
        filename: 文件名。
        chunks: 切片结果列表。

    Returns:
        None。

    Raises:
        无。
    """
    print(f"[chunk-debug] filename={filename}, chunk_count={len(chunks)}")
    for chunk in chunks:
        chunk_text = chunk.text.strip()
        print(
            "[chunk-debug] "
            f"char_count={chunk.stats.char_count}, "
            f"text={chunk_text}"
        )


def _split_parsed_text(
        text: str,
        chunk_strategy: ChunkStrategyType,
        chunk_size: int,
) -> list[SplitChunk]:
    """
    对解析后的文本执行切片。

    Args:
        text: 完整文本。
        chunk_strategy: 切片策略。
        chunk_size: 生效切片大小。

    Returns:
        切片结果列表；空白文本返回空列表。

    Raises:
        ServiceException: 切片策略不支持或切片依赖不可用。
    """
    config = SplitConfig(
        chunk_size=chunk_size,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )
    return split_text(text, chunk_strategy, config) if text.strip() else []


def delete_document(knowledge_name: str, document_id: int) -> None:
    """
    删除文档在知识库中的全部切片。

    Args:
        knowledge_name: 知识库名称。
        document_id: 文档 ID。

    Returns:
        None。

    Raises:
        ServiceException: 知识库不存在或删除失败。
    """
    vector_repository.ensure_collection_exists(knowledge_name=knowledge_name)
    vector_repository.delete_document_chunks(
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
    分页查询文档切片。

    Args:
        knowledge_name: 知识库名称。
        document_id: 文档 ID。
        page_num: 页码（从 1 开始）。
        page_size: 每页条数。

    Returns:
        `(rows, total)`，分别为当前页数据与总数。

    Raises:
        ServiceException: 分页参数非法、知识库不存在或查询失败。
    """
    if page_num <= 0 or page_size <= 0:
        raise ServiceException(
            code=ResponseCode.BAD_REQUEST,
            message="page_num 和 page_size 必须大于 0",
        )
    vector_repository.ensure_collection_exists(knowledge_name=knowledge_name)
    return vector_repository.list_document_chunks(
        knowledge_name=knowledge_name,
        document_id=document_id,
        page_num=page_num,
        page_size=page_size,
    )
