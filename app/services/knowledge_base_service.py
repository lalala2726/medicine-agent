import hashlib
import os
from collections.abc import Callable
from pathlib import Path
from time import time

from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException
from app.core.llms import create_embedding_model
from app.core.mq.contracts.models import ProcessingStageDetail
from app.core.mq.observability.import_logger import ImportStage, import_log
from app.rag.chunking import ChunkStrategyType, SplitChunk, SplitConfig, split_text
from app.rag.file_loader import parse_downloaded_file, validate_url_extension
from app.repositories import vector_repository
from app.schemas.knowledge_import import (
    ImportChunk,
    ImportSingleFileFailedResult,
    ImportSingleFileResult,
    ImportSingleFileSuccessResult,
)
from app.utils.file_utils import FileUtils
from app.utils.token_utills import TokenUtils

DEFAULT_CHUNK_SIZE = 500  # 默认切片长度（字符）
DEFAULT_TOKEN_SIZE = 100  # 默认 token 切片长度
DEFAULT_CHUNK_OVERLAP = 50  # 默认切片重叠长度（字符）
EMBED_MAX_TOKEN_SIZE = 8192  # 向量化前单文本最大 token 限制
DEFAULT_VECTOR_BATCH_SIZE = 20  # 向量化与入库默认批次大小
ProcessingStageCallback = Callable[[ProcessingStageDetail], None]


def _split_batches(items: list, batch_size: int) -> list[list]:
    """
    将列表按批次大小切分。

    Args:
        items: 原始列表。
        batch_size: 单批最大条数。

    Returns:
        二维列表，每个子列表为一个批次。

    Raises:
        无。
    """
    result: list[list] = []
    for index in range(0, len(items), batch_size):
        result.append(items[index:index + batch_size])
    return result


def _resolve_vector_batch_size() -> int:
    """
    读取向量处理批次配置。

    Args:
        无。

    Returns:
        向量处理批次大小。

    Raises:
        ServiceException: 配置值不是正整数时抛出。
    """
    raw_value = (os.getenv("KNOWLEDGE_VECTOR_BATCH_SIZE") or "").strip()
    if not raw_value:
        return DEFAULT_VECTOR_BATCH_SIZE
    try:
        parsed = int(raw_value)
    except ValueError as exc:
        raise ServiceException(
            code=ResponseCode.INTERNAL_ERROR,
            message="KNOWLEDGE_VECTOR_BATCH_SIZE 必须是正整数",
        ) from exc
    if parsed <= 0:
        raise ServiceException(
            code=ResponseCode.INTERNAL_ERROR,
            message="KNOWLEDGE_VECTOR_BATCH_SIZE 必须大于 0",
        )
    return parsed


def _build_source_hash(text: str) -> str:
    """
    计算文本 sha256 哈希。

    Args:
        text: 原始文本。

    Returns:
        十六进制哈希字符串。

    Raises:
        无。
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _create_embedding_client(*, embedding_model: str, embedding_dim: int):
    """
    创建向量模型客户端。

    Args:
        embedding_model: 向量模型名称。
        embedding_dim: 向量维度。

    Returns:
        向量模型客户端实例。

    Raises:
        ServiceException: 模型初始化失败时抛出。
    """
    try:
        return create_embedding_model(
            model=embedding_model,
            dimensions=embedding_dim,
        )
    except Exception as exc:  # pragma: no cover - 依赖外部模型 SDK
        raise ServiceException(message=f"初始化向量模型失败: {exc}") from exc


def embed_texts(
        texts: list[str],
        *,
        embedding_client,
) -> list[list[float]]:
    """
    对文本列表执行向量化。

    Args:
        texts: 待向量化文本列表。
        embedding_client: 向量模型客户端。

    Returns:
        向量列表，顺序与 `texts` 一致。

    Raises:
        ServiceException: 文本超过 token 限制或向量化失败。
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

    try:
        return embedding_client.embed_documents(texts)
    except Exception as exc:  # pragma: no cover - 依赖外部模型 SDK
        raise ServiceException(message=f"嵌入文本失败: {exc}") from exc


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


def load_collection_state(knowledge_name: str) -> dict:
    """
    启用知识库对应的 Milvus collection（加载到查询节点）。

    Args:
        knowledge_name: 知识库名称。

    Returns:
        dict: 含 `knowledge_name` 与 `load_state`（可选 `progress`）。

    Raises:
        ServiceException: collection 不存在或加载失败。
    """
    return vector_repository.load_collection_state(knowledge_name=knowledge_name)


def release_collection_state(knowledge_name: str) -> dict:
    """
    关闭知识库对应的 Milvus collection（从查询节点释放）。

    Args:
        knowledge_name: 知识库名称。

    Returns:
        dict: 含 `knowledge_name` 与 `load_state`（可选 `progress`）。

    Raises:
        ServiceException: collection 不存在或释放失败。
    """
    return vector_repository.release_collection_state(knowledge_name=knowledge_name)


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


def _notify_processing_stage(
        callback: ProcessingStageCallback | None,
        detail: ProcessingStageDetail,
) -> None:
    """安全触发一次处理阶段回调。

    Args:
        callback: 调用方提供的可选回调函数。
        detail: 当前处理步骤对应的阶段枚举。

    Returns:
        None。
    """
    if callback is None:
        return
    try:
        callback(detail)
    except Exception:
        # 阶段回调失败不应影响主导入流程。
        return


def import_single_file(
        url: str,
        knowledge_name: str,
        document_id: int,
        embedding_model: str,
        chunk_strategy: ChunkStrategyType = ChunkStrategyType.CHARACTER,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        token_size: int = DEFAULT_TOKEN_SIZE,
        task_uuid: str = "-",
        on_processing_stage: ProcessingStageCallback | None = None,
) -> ImportSingleFileResult:
    """
    执行单个文件的完整导入流程：校验 → 下载 → 解析 → 切片 → 向量化 → 入库。

    每个关键阶段输出结构化日志，调用方只需关注返回值。

    Args:
        url: 远程文件 URL。
        knowledge_name: 知识库名称。
        document_id: 文档 ID。
        embedding_model: 向量模型名称。
        chunk_strategy: 切片策略。
        chunk_size: 字符切片大小。
        token_size: token 切片大小。
        task_uuid: 导入任务唯一标识（用于日志关联），默认 ``"-"``。
        on_processing_stage: 可选阶段回调，按下载/解析/切片/向量化/入库触发。

    Returns:
        成功时返回 `ImportSingleFileSuccessResult`；
        失败时返回 `ImportSingleFileFailedResult`。

    Raises:
        ServiceException: 参数校验失败时直接抛出（非可恢复错误）。
    """
    vector_batch_size = _resolve_vector_batch_size()
    filename: str | None = None
    embedding_dim = 0

    try:
        # 步骤 1：URL 后缀校验
        source_extension = validate_url_extension(url)

        # 步骤 2：知识库与模型准备
        vector_repository.ensure_collection_exists(knowledge_name=knowledge_name)
        embedding_dim = vector_repository.get_collection_embedding_dim(
            knowledge_name=knowledge_name
        )
        embedding_client = _create_embedding_client(
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
        )

        # 步骤 3：下载
        _notify_processing_stage(on_processing_stage, ProcessingStageDetail.DOWNLOADING)
        import_log(ImportStage.DOWNLOAD_START, task_uuid, url=url)
        filename, file_path = _download_file(url)
        _validate_file_not_empty(file_path)
        file_size = file_path.stat().st_size
        import_log(ImportStage.DOWNLOAD_DONE, task_uuid, filename=filename, size=file_size)

        # 步骤 4：解析
        _notify_processing_stage(on_processing_stage, ProcessingStageDetail.PARSING)
        parsed_document = parse_downloaded_file(
            file_path=file_path,
            source_url=url,
        )
        parsed_text = parsed_document.text or ""
        import_log(
            ImportStage.PARSE_DONE,
            task_uuid,
            filename=filename,
            file_kind=parsed_document.file_kind.value,
            text_length=len(parsed_text),
        )

        # 步骤 5：切片
        _notify_processing_stage(on_processing_stage, ProcessingStageDetail.CHUNKING)
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
        import_log(
            ImportStage.CHUNK_DONE,
            task_uuid,
            chunk_count=len(chunks),
            strategy=chunk_strategy.value,
            chunk_size=effective_chunk_size,
        )

        # 步骤 6：分批向量化并入库
        source_hash = _build_source_hash(parsed_text)
        vector_count = 0
        insert_batches = 0
        batch_chunks_list = _split_batches(chunks, vector_batch_size)
        total_batches = len(batch_chunks_list)
        if batch_chunks_list:
            _notify_processing_stage(on_processing_stage, ProcessingStageDetail.EMBEDDING)
            _notify_processing_stage(on_processing_stage, ProcessingStageDetail.INSERTING)

        for batch_index, batch_chunks in enumerate(batch_chunks_list, start=1):
            batch_texts = [chunk.text for chunk in batch_chunks]
            batch_embeddings = embed_texts(
                batch_texts,
                embedding_client=embedding_client,
            )
            import_log(
                ImportStage.EMBED_BATCH,
                task_uuid,
                batch=f"{batch_index}/{total_batches}",
                texts=len(batch_texts),
            )

            char_counts = [chunk.stats.char_count for chunk in batch_chunks]
            vector_repository.insert_embeddings(
                knowledge_name=knowledge_name,
                document_id=document_id,
                embeddings=batch_embeddings,
                texts=batch_texts,
                start_chunk_no=vector_count + 1,
                chunk_strategy=chunk_strategy.name.lower(),
                chunk_size=chunk_size,
                token_size=token_size,
                source_hash=source_hash,
                char_counts=char_counts,
                created_at_ts=int(time() * 1000),
            )
            vector_count += len(batch_chunks)
            insert_batches += 1

        import_log(
            ImportStage.INSERT_DONE,
            task_uuid,
            vector_count=vector_count,
            insert_batches=insert_batches,
        )
        import_log(
            ImportStage.COMPLETED,
            task_uuid,
            filename=filename,
            chunk_count=len(chunks),
            vector_count=vector_count,
        )

        return ImportSingleFileSuccessResult(
            file_url=url,
            filename=filename,
            source_extension=source_extension,
            file_kind=parsed_document.file_kind.value,
            mime_type=parsed_document.mime_type,
            text_length=len(parsed_text),
            chunk_count=len(chunks),
            vector_count=vector_count,
            insert_batches=insert_batches,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            chunks=[ImportChunk.from_split_chunk(chunk) for chunk in chunks],
        )
    except Exception as exc:
        import_log(
            ImportStage.FAILED,
            task_uuid,
            url=url,
            filename=filename,
            error=str(exc),
        )
        return ImportSingleFileFailedResult(
            file_url=url,
            filename=filename,
            error=str(exc),
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
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


def _print_chunks_to_console(*, filename: str, chunks: list[SplitChunk]) -> None:
    """
    保留切片打印钩子占位实现。

    Args:
        filename: 文件名。
        chunks: 切片结果列表。

    Returns:
        None。

    Raises:
        无。
    """
    return None


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
