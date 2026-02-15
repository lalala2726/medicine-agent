"""
图像解析 API 路由

接收图片 URL 并调用大模型进行结构化解析。
"""

from typing import List

from fastapi import APIRouter
from pydantic import AliasChoices, BaseModel, Field

from app.core.codes import ResponseCode
from app.core.exceptions import ServiceException
from app.schemas.response import ApiResponse
from app.services.image_parse_service import parse_drug_images

router = APIRouter(prefix="/image_parse", tags=["图像解析"])


class ImageParseRequest(BaseModel):
    """图像解析请求参数"""

    image_urls: List[str] = Field(
        ...,
        min_length=1,
        max_length=5,
        validation_alias=AliasChoices("image_urls", "images"),
        description="图片 URL 列表，最少1张，最多5张",
    )


@router.post("/drug", summary="解析药品图片")
async def parse_image(request: ImageParseRequest) -> ApiResponse[dict]:
    """
    接收图片 URL 并使用大模型进行解析

    Args:
        request: 图像解析请求参数

    Returns:
        ApiResponse[dict]: 解析结果

    Raises:
        ServiceException: 图片列表为空时抛出异常
    """

    data = parse_drug_images(request.image_urls)
    return ApiResponse.success(data=data, message="解析成功")
