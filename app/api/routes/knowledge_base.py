from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

from app.schemas.response import ApiResponse
from app.services.knowledge_base_service import create_collection, delete_collection, import_knowledge_service
from core.exceptions import ServiceException

router = APIRouter(prefix="/knowledge_base", tags=["知识库管理"])


class CreateCollectionRequest(BaseModel):
    """创建向量数据库 collection 的请求参数。"""

    collection_name: str = Field(
        ...,
        min_length=1,
        pattern=r"^[A-Za-z][A-Za-z0-9_]*$",
        description="collection 名称（仅英文/数字/下划线，需以字母开头）",
    )
    embedding_dim: int = Field(..., gt=0, description="向量维度")
    description: Optional[str] = Field(default="", description="collection 描述")

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

    collection_name: str = Field(
        ...,
        min_length=1,
        pattern=r"^[A-Za-z][A-Za-z0-9_]*$",
        description="collection 名称（仅英文/数字/下划线，需以字母开头）",
    )
    file_urls: list[str] = Field(..., description="导入文件的URL")


@router.post(path="", summary="创建向量库")
async def create_knowledge_base(request: CreateCollectionRequest) -> ApiResponse[dict]:
    create_collection(request.collection_name, request.embedding_dim, request.description or "")
    return ApiResponse.success(
        data={"collection_name": request.collection_name},
        message="创建成功",
    )


@router.delete("/{id}", summary="删除向量库")
async def delete_knowledge_base(id: int) -> ApiResponse[dict]:
    collection_name = str(id)
    delete_collection(collection_name)
    return ApiResponse.success(
        data={"collection_name": collection_name},
        message="删除成功",
    )


@router.post(path="/import", summary="导入知识库")
async def import_knowledge(request: ImportKnowledgeRequest):
    import_knowledge_service(request.collection_name, request.file_urls)
    return ApiResponse.success("开始导入数据～ 完整后将通过回调通知您！")
