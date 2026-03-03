from pathlib import Path

from loguru import logger

from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException
from app.core.mq.models import KnowledgeImportMessage
from app.core.mq.publisher import publish_import_messages
from app.rag.chunking import ChunkStrategyType, SplitChunk, SplitConfig, split_text
from app.rag.file_loader import parse_downloaded_file, validate_url_extension
from app.services import vector_service
from app.utils.file_utils import FileUtils

DEFAULT_CHUNK_SIZE = 500  # 默认切片长度（字符）
DEFAULT_TOKEN_SIZE = 100  # 默认 token 切片长度
DEFAULT_CHUNK_OVERLAP = 50  # 默认切片重叠长度（字符）


async def submit_import_to_queue(
        knowledge_name: str,
        document_id: int,
        file_url: list[str] | str,
        chunk_strategy: ChunkStrategyType = ChunkStrategyType.CHARACTER,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        token_size: int = DEFAULT_TOKEN_SIZE,
) -> dict:
    """
    功能描述:
        将导入请求转换为 RabbitMQ 消息并异步投递，不在请求线程中执行解析与切片。

    参数说明:
        knowledge_name (str): 知识库名称。
        document_id (int): 文档 ID。
        file_url (list[str] | str): 待导入文件 URL 列表或单个 URL 字符串。
        chunk_strategy (ChunkStrategyType): 切片策略类型，默认值为 ChunkStrategyType.CHARACTER。
        chunk_size (int): 字符类切片大小，默认值为 DEFAULT_CHUNK_SIZE。
        token_size (int): token 切片大小，默认值为 DEFAULT_TOKEN_SIZE。

    返回值:
        dict: 投递结果，包含 accepted_count 与 task_uuids。

    异常说明:
        ServiceException:
            - URL 参数非法或为空时抛出；
            - URL 后缀不支持时抛出；
            - RabbitMQ 配置缺失或消息投递失败时由下游抛出。
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


def _download_file(url: str) -> tuple[str, Path]:
    """
    功能描述:
        下载文件并返回（文件名，固定下载目录下的文件路径）。

    参数说明:
        url (str): 远程文件 URL。

    返回值:
        tuple[str, Path]:
            - 第 1 项: 解析后的文件名；
            - 第 2 项: 固定目录中的落盘路径。

    异常说明:
        ServiceException: 下载失败或下载目录配置异常时由下游抛出。

    说明:
        单独封装该函数是为了测试可替换（monkeypatch），
        同时避免导入流程与底层下载实现强耦合。
    """

    return FileUtils.download_file(url)


def create_collection(
        knowledge_name: str, embedding_dim: int, description: str
) -> None:
    """
    功能描述:
        创建 Milvus 知识库并应用业务字段 schema。

    参数说明:
        knowledge_name (str): knowledge 名称。
        embedding_dim (int): 向量维度。
        description (str): knowledge 描述。

    返回值:
        None: 创建完成无返回值。

    异常说明:
        ServiceException: 集合已存在、参数非法或底层创建失败时由下游抛出。
    """
    vector_service.create_collection(knowledge_name, embedding_dim, description)


def delete_knowledge(knowledge_name: str) -> None:
    """
    功能描述:
        删除 Milvus 知识库。

    参数说明:
        knowledge_name (str): knowledge 名称。

    返回值:
        None: 删除完成无返回值。

    异常说明:
        ServiceException: 知识库不存在或底层删除失败时由下游抛出。
    """
    vector_service.delete_collection(knowledge_name)


def _validate_file_not_empty(file_path: Path) -> None:
    """
    功能描述:
        校验下载文件大小，防止空文件进入解析与切片流程。

    参数说明:
        file_path (Path): 下载后的本地文件路径。

    返回值:
        None: 校验通过无返回值。

    异常说明:
        ServiceException: 文件大小为 0 时抛出。
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
            f"char_count={chunk.stats.char_count}, "
            f"text={chunk_text}"
        )


def _split_parsed_text(
        text: str,
        chunk_strategy: ChunkStrategyType,
        chunk_size: int,
) -> list[SplitChunk]:
    """
    功能描述:
        对解析后的单一文本执行切片并返回统一切片结果。

    参数说明:
        text (str): 解析后的完整文本。
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
    return split_text(text, chunk_strategy, config) if text.strip() else []


def delete_document(knowledge_name: str, document_id: int) -> None:
    """
    功能描述:
        删除指定文档在向量库中的全部切片数据。

    参数说明:
        knowledge_name (str): 知识库名称。
        document_id (int): 文档 ID。

    返回值:
        None: 删除完成无返回值。

    异常说明:
        ServiceException: 知识库不存在或删除执行失败时由下游抛出。
    """
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
    功能描述:
        分页查询知识库中的文档切片数据。

    参数说明:
        knowledge_name (str): 知识库名。
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
            - 知识库不存在或查询失败时由下游抛出。
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
