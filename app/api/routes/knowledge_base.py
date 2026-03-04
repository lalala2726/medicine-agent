from typing import Optional

from fastapi import APIRouter, Depends, Path, Query
from pydantic import BaseModel, Field, field_validator

from app.core.exception.exceptions import ServiceException
from app.rag.chunking import ChunkStrategyType
from app.schemas.response import ApiResponse, PageResponse
from app.services.knowledge_base_service import (
    create_collection,
    delete_document,
    delete_knowledge,
    load_collection_state,
    list_knowledge_chunks,
    release_collection_state,
    submit_import_to_queue,
)

router = APIRouter(prefix="/knowledge_base", tags=["知识库管理"])


class CreateCollectionRequest(BaseModel):
    """创建知识库请求参数"""
    knowledge_name: str = Field(
        ...,
        pattern=r"^[A-Za-z][A-Za-z0-9_]*$",
        description="知识库名称（英文/数字/下划线，字母开头）"
    )
    embedding_dim: int = Field(..., gt=0, description="向量维度")
    description: Optional[str] = Field(default="", description="知识库描述")

    @field_validator("embedding_dim")
    @classmethod
    def validate_embedding_dim(cls, value: int) -> int:
        """验证向量维度"""
        if value < 128 or value > 4096:
            raise ServiceException("向量维度必须在 128 到 4096 之间")
        if value % 2 != 0:
            raise ServiceException("向量维度必须能被 2 整除")
        return value


class ImportKnowledgeRequest(BaseModel):
    """导入知识库请求参数"""
    knowledge_name: str = Field(
        ...,
        pattern=r"^[A-Za-z][A-Za-z0-9_]*$",
        description="知识库名称"
    )
    document_id: int = Field(..., gt=0, description="文档ID")
    file_urls: list[str] = Field(..., description="导入文件的URL")
    embedding_model: str = Field(..., min_length=1, description="向量模型名称")
    chunk_strategy: ChunkStrategyType = Field(
        default=ChunkStrategyType.CHARACTER,
        description="切片策略：character/recursive/token/markdown_header",
    )
    chunk_size: int = Field(default=500, ge=1, le=10000, description="切片大小（字符）")
    token_size: int = Field(default=100, ge=1, le=1000, description="token大小")


@router.post(path="", summary="创建知识库")
async def create_knowledge_base(request: CreateCollectionRequest) -> ApiResponse[dict]:
    """
    创建知识库

    Args:
        request: 创建知识库请求参数

    Returns:
        ApiResponse[dict]: 创建成功响应
    """
    create_collection(
        request.knowledge_name,
        request.embedding_dim,
        request.description or "",
    )
    return ApiResponse.success(
        data={"knowledge_name": request.knowledge_name},
        message="创建成功",
    )


class DeleteKnowledgeRequest(BaseModel):
    """删除知识库请求参数"""
    knowledge_name: str = Field(
        ..., pattern=r"^[A-Za-z][A-Za-z0-9_]*$", description="知识库名称"
    )


@router.delete("", summary="删除知识库")
async def delete_knowledge_base(
        request: DeleteKnowledgeRequest,
) -> ApiResponse[dict]:
    """
    删除知识库

    Args:
        request: 删除知识库请求参数

    Returns:
        ApiResponse[dict]: 删除成功响应
    """
    delete_knowledge(request.knowledge_name)
    return ApiResponse.success(
        data={"knowledge_name": request.knowledge_name},
        message="删除成功",
    )


class KnowledgeLoadRequest(BaseModel):
    """启用/关闭集合请求参数"""
    knowledge_name: str = Field(
        ...,
        pattern=r"^[A-Za-z][A-Za-z0-9_]*$",
        description="知识库名称",
    )


@router.post(path="/load", summary="启用知识库")
async def load_knowledge_base(
        request: KnowledgeLoadRequest,
) -> ApiResponse[dict]:
    """
    启用知识库对应集合（load collection）。

    Args:
        request: 启用请求参数。

    Returns:
        ApiResponse[dict]: 启用成功响应。
    """
    result = load_collection_state(request.knowledge_name)
    return ApiResponse.success(
        data=result,
        message="启用成功",
    )


@router.post(path="/release", summary="关闭知识库")
async def release_knowledge_base(
        request: KnowledgeLoadRequest,
) -> ApiResponse[dict]:
    """
    关闭知识库对应集合（release collection）。

    Args:
        request: 关闭请求参数。

    Returns:
        ApiResponse[dict]: 关闭成功响应。
    """
    result = release_collection_state(request.knowledge_name)
    return ApiResponse.success(
        data=result,
        message="关闭成功",
    )


class ListDocumentChunksRequest(BaseModel):
    """分页查询文档切片请求参数"""
    knowledge_name: str = Field(
        ..., pattern=r"^[A-Za-z][A-Za-z0-9_]*$", description="知识库名称"
    )
    document_id: int = Field(..., gt=0, description="文档ID")
    page: int = Field(default=1, gt=0, description="页码")
    page_size: int = Field(default=10, ge=1, le=100, description="每页数量")


@router.get("/document/chunks/list", summary="分页查询文档切片")
async def list_document_chunks(
        request: ListDocumentChunksRequest = Depends(),
) -> ApiResponse[PageResponse[dict]]:
    """
    分页查询文档切片

    Args:
        request: 分页查询请求参数

    Returns:
        ApiResponse[PageResponse[dict]]: 分页响应数据
    """
    rows, total = list_knowledge_chunks(
        knowledge_name=request.knowledge_name,
        document_id=request.document_id,
        page_num=request.page,
        page_size=request.page_size,
    )
    return ApiResponse.page(
        rows=rows,
        total=total,
        page_num=request.page,
        page_size=request.page_size,
    )


@router.delete("/document/{id}", summary="删除文档切片")
async def delete_document_chunk(
        id: int = Path(..., gt=0, description="文档ID"),
        knowledge_name: str = Query(
            ...,
            min_length=1,
            pattern=r"^[A-Za-z][A-Za-z0-9_]*$",
            description="知识库名称",
        ),
) -> ApiResponse[dict]:
    """
    删除文档，当删除文档之后相关知识库中的文档切片也会被删除

    Args:
        id: 文档ID
        knowledge_name: 知识库名称

    Returns:
        ApiResponse[dict]: 删除成功响应
    """
    delete_document(
        knowledge_name=knowledge_name,
        document_id=id,
    )
    return ApiResponse.success(
        data={"id": id},
        message="删除成功",
    )


@router.post(path="/document/import", summary="导入知识库")
async def import_knowledge(
        request: ImportKnowledgeRequest,
) -> ApiResponse[str]:
    """
    功能描述:
        接收导入请求并提交到 MQ 异步队列，立即返回受理结果。

    参数说明:
        request (ImportKnowledgeRequest): 导入请求体，包含知识库、文档与切片参数。

    返回值:
        ApiResponse[str]: 受理成功响应，不等待导入处理完成。

    异常说明:
        ServiceException: 参数校验失败或 MQ 提交失败时由下游抛出。
    """
    await submit_import_to_queue(
        knowledge_name=request.knowledge_name,
        document_id=request.document_id,
        file_url=request.file_urls,
        embedding_model=request.embedding_model,
        chunk_strategy=request.chunk_strategy,
        chunk_size=request.chunk_size,
        token_size=request.token_size,
    )
    return ApiResponse.success("已接收导入请求，正在异步队列处理中～")
