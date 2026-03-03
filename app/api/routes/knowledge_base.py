from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Path, Query
from pydantic import BaseModel, Field, field_validator

from app.core.exception.exceptions import ServiceException
from app.rag.chunking import ChunkStrategyType
from app.schemas.response import ApiResponse, PageResponse
from app.services.knowledge_base_service import (
    create_collection,
    delete_document,
    delete_knowledge,
    import_knowledge_service,
    list_knowledge_chunks,
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
    create_collection(request.knowledge_name, request.embedding_dim, request.description or "")
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
        background_tasks: BackgroundTasks
) -> ApiResponse[str]:
    """
    导入知识库（后台异步处理）

    Args:
        request: 导入知识库请求参数
        background_tasks: 后台任务管理器

    Returns:
        ApiResponse[str]: 导入请求已接收响应
    """
    background_tasks.add_task(
        import_knowledge_service,
        request.knowledge_name,
        request.document_id,
        request.file_urls,
        request.chunk_strategy,
        request.chunk_size,
        request.token_size,
    )
    return ApiResponse.success("已接收导入请求，正在后台处理中～")
