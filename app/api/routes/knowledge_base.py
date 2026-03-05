from typing import Optional

from fastapi import APIRouter, Depends, Path, Query
from pydantic import BaseModel, Field, field_validator

from app.core.exception.exceptions import ServiceException
from app.core.security import allow_system
from app.schemas.response import ApiResponse
from app.services.knowledge_base_service import (
    create_collection,
    delete_document,
    delete_knowledge,
    load_collection_state,
    list_knowledge_chunks,
    release_collection_state,
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


@router.post(path="", summary="创建知识库")
@allow_system
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
@allow_system
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
@allow_system
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
@allow_system
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
    page_size: int = Field(default=50, ge=1, le=100, description="每页数量")


class DocumentChunksPageResponse(BaseModel):
    """文档切片分页响应数据"""

    rows: list[dict] = Field(..., description="当前页数据列表")
    total: int = Field(..., description="数据总数")
    page_num: int = Field(..., description="当前页码")
    page_size: int = Field(..., description="每页数量")
    has_next: bool = Field(..., description="是否存在下一页")


@router.get("/document/chunks/list", summary="分页查询文档切片")
@allow_system
async def list_document_chunks(
        request: ListDocumentChunksRequest = Depends(),
) -> ApiResponse[DocumentChunksPageResponse]:
    """
    分页查询文档切片

    Args:
        request: 分页查询请求参数

    Returns:
        ApiResponse[DocumentChunksPageResponse]: 分页响应数据
    """
    rows, total = list_knowledge_chunks(
        knowledge_name=request.knowledge_name,
        document_id=request.document_id,
        page_num=request.page,
        page_size=request.page_size,
    )
    has_next = (request.page * request.page_size) < total
    return ApiResponse.success(
        data=DocumentChunksPageResponse(
            rows=rows,
            total=total,
            page_num=request.page,
            page_size=request.page_size,
            has_next=has_next,
        )
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
