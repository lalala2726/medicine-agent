from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException
from app.repositories import vector_repository


def create_collection(
        knowledge_name: str,
        embedding_dim: int,
        description: str,
) -> None:
    """创建知识库对应的 Milvus collection。

    Args:
        knowledge_name: 知识库名称。
        embedding_dim: 向量维度。
        description: 知识库描述。

    Raises:
        ServiceException: collection 已存在或创建失败时抛出。
    """
    vector_repository.create_collection(
        knowledge_name=knowledge_name,
        embedding_dim=embedding_dim,
        description=description,
    )


def delete_knowledge(knowledge_name: str) -> None:
    """删除知识库对应的 Milvus collection。

    Args:
        knowledge_name: 知识库名称。

    Raises:
        ServiceException: collection 不存在或删除失败时抛出。
    """
    vector_repository.delete_collection(knowledge_name=knowledge_name)


def load_collection_state(knowledge_name: str) -> dict:
    """启用知识库对应的 Milvus collection。

    Args:
        knowledge_name: 知识库名称。

    Returns:
        dict: 包含 ``knowledge_name`` 与 ``load_state`` 的状态结果。

    Raises:
        ServiceException: collection 不存在或加载失败时抛出。
    """
    return vector_repository.load_collection_state(knowledge_name=knowledge_name)


def release_collection_state(knowledge_name: str) -> dict:
    """关闭知识库对应的 Milvus collection。

    Args:
        knowledge_name: 知识库名称。

    Returns:
        dict: 包含 ``knowledge_name`` 与 ``load_state`` 的状态结果。

    Raises:
        ServiceException: collection 不存在或释放失败时抛出。
    """
    return vector_repository.release_collection_state(knowledge_name=knowledge_name)


def delete_documents(knowledge_name: str, document_ids: list[int]) -> None:
    """批量删除文档在知识库中的全部切片。

    Args:
        knowledge_name: 知识库名称。
        document_ids: 文档 ID 列表。

    Raises:
        ServiceException: 知识库不存在或删除失败时抛出。
    """
    vector_repository.ensure_collection_exists(knowledge_name=knowledge_name)
    vector_repository.delete_document_chunks(
        knowledge_name=knowledge_name,
        document_ids=document_ids,
    )


def update_document_status(
        knowledge_name: str,
        primary_id: int,
        status: int,
) -> None:
    """按向量主键更新文档切片状态。

    Args:
        knowledge_name: 知识库名称。
        primary_id: 向量数据库主键 ID。
        status: 状态值，仅允许 0 或 1。

    Raises:
        ServiceException: 知识库不存在、状态非法或更新失败时抛出。
    """
    vector_repository.ensure_collection_exists(knowledge_name=knowledge_name)
    vector_repository.update_document_chunk_status(
        knowledge_name=knowledge_name,
        primary_id=primary_id,
        status=status,
    )


def list_knowledge_chunks(
        knowledge_name: str,
        document_id: int,
        page_num: int,
        page_size: int,
) -> tuple[list[dict], int]:
    """分页查询文档切片。

    Args:
        knowledge_name: 知识库名称。
        document_id: 文档 ID。
        page_num: 页码，从 1 开始。
        page_size: 每页条数。

    Returns:
        tuple[list[dict], int]: 当前页数据与总数。

    Raises:
        ServiceException: 分页参数非法、知识库不存在或查询失败时抛出。
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
