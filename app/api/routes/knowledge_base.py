from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Path
from pydantic import BaseModel, Field, field_validator

from app.core.chunking import ChunkStrategyType
from app.core.exceptions import ServiceException
from app.schemas.response import ApiResponse, PageResponse
from app.services.knowledge_base_service import (
    create_collection,
    delete_knowledge,
    import_knowledge_service,
    list_knowledge_chunks,
)

router = APIRouter(prefix="/knowledge_base", tags=["知识库管理"])


class CreateCollectionRequest(BaseModel):
    """创建知识库的请求参数。"""

    knowledge_name: str = Field(
        ...,
        min_length=1,
        pattern=r"^[A-Za-z][A-Za-z0-9_]*$",
        description="knowledge 名称（仅英文/数字/下划线，需以字母开头）",
    )
    embedding_dim: int = Field(..., gt=0, description="向量维度")
    description: Optional[str] = Field(default="", description="knowledge 描述")

    @field_validator("embedding_dim")
    @classmethod
    def validate_embedding_dim(cls, value: int) -> int:
        min_dim = 128
        max_dim = 4096
        if value < min_dim or value > max_dim:
            raise ServiceException("向量维度必须在 128 到 4096 之间")
        if value % 2 != 0:
            raise ServiceException("向量维度必须能被 2 整除")
        return value


class ImportKnowledgeRequest(BaseModel):
    """导入知识库的请求参数。"""

    knowledge_name: str = Field(
        ...,
        min_length=1,
        pattern=r"^[A-Za-z][A-Za-z0-9_]*$",
        description="knowledge 名称（仅英文/数字/下划线，需以字母开头）",
    )
    document_id: int = Field(..., gt=0, description="文档ID")
    file_urls: list[str] = Field(..., description="导入文件的URL")
    chunk_strategy: ChunkStrategyType = Field(
        default=ChunkStrategyType.LENGTH, description="切片类型"
    )
    chunk_size: int = Field(
        default=500, ge=1, le=10000, description="切片大小（1-10000字符）"
    )
    token_size: int = Field(
        default=100, ge=1, le=1000, description="token 大小（1-1000token）"
    )
    parse_images: bool = Field(default=False, description="是否解析图片")


@router.post(path="", summary="创建知识库")
async def create_knowledge_base(request: CreateCollectionRequest) -> ApiResponse[dict]:
    create_collection(request.knowledge_name, request.embedding_dim, request.description or "")
    return ApiResponse.success(
        data={"knowledge_name": request.knowledge_name},
        message="创建成功",
    )


@router.delete("/{knowledge_name}", summary="删除知识库")
async def delete_knowledge_base(
        knowledge_name: str = Path(
            ...,
            min_length=1,
            pattern=r"^[A-Za-z][A-Za-z0-9_]*$",
            description="knowledge 名称（仅英文/数字/下划线，需以字母开头）",
        )
) -> ApiResponse[dict]:
    delete_knowledge(knowledge_name)
    return ApiResponse.success(
        data={"knowledge_name": knowledge_name},
        message="删除成功",
    )


class ListDocumentChunksRequest(BaseModel):
    """分页查询文档切片请求参数"""
    knowledge_name: str = Field(
        ...,
        min_length=1,
        pattern=r"^[A-Za-z][A-Za-z0-9_]*$",
        description="knowledge 名称（仅英文/数字/下划线，需以字母开头）",
    )
    document_id: int = Field(..., gt=0, description="文档ID")
    page: int = Field(default=1, gt=0, description="页码")
    page_size: int = Field(default=10, ge=1, le=100, description="每页数量")


@router.get(
    "/chunks/list",
    summary="分页查询文档切片",
)
async def list_document_chunks(
        request: ListDocumentChunksRequest = Depends(),
) -> ApiResponse[PageResponse[dict]]:
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


@router.post(path="/import", summary="导入知识库")
async def import_knowledge(
        request: ImportKnowledgeRequest, background_tasks: BackgroundTasks
):
    background_tasks.add_task(
        import_knowledge_service,
        request.knowledge_name,
        request.document_id,
        request.file_urls,
        request.chunk_strategy,
        request.chunk_size,
        request.token_size,
        request.parse_images,
    )
    return ApiResponse.success("已接收导入请求，正在后台处理中～")
