"""
图像解析 API 路由

接收 Spring Boot 处理好的图片 base64 并调用大模型进行结构化解析。
"""

from typing import List

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.codes import ResponseCode
from app.core.exceptions import ServiceException
from app.schemas.response import ApiResponse
from app.services.image_parse_service import parse_drug_images

router = APIRouter(prefix="/image/parse", tags=["图像解析"])


class ImageParseRequest(BaseModel):
    """图像解析请求参数"""

    images: List[str] = Field(..., description="图片 base64 编码列表")


@router.post("", summary="解析药品图片")
async def parse_image(request: ImageParseRequest) -> ApiResponse[dict]:
    """接收 base64 图片并使用大模型进行解析。"""

    if not request.images:
        raise ServiceException(code=ResponseCode.BAD_REQUEST, message="图片不能为空")

    data = parse_drug_images(request.images)
    return ApiResponse.success(data=data, message="解析成功")
