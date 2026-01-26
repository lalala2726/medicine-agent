"""
图像解析 API 路由

接收 Spring Boot 处理好的图片 base64 并调用大模型进行结构化解析。
"""

import json
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm import get_chat_model
from app.core.prompts import DRUG_PARSER_PROMPT
from app.schemas.response import ApiResponse

router = APIRouter(prefix="/image/parse", tags=["图像解析"])


class ImageParseRequest(BaseModel):
    """图像解析请求参数"""

    images: List[str] = Field(..., description="图片 base64 编码列表")


@router.post("/drug", summary="解析药品图片")
async def parse_image(request: ImageParseRequest) -> ApiResponse[dict]:
    """接收 base64 图片并使用大模型进行解析。"""

    def normalize_image_data(image_str: str) -> str:
        if image_str.startswith("data:image"):
            return image_str
        return f"data:image/png;base64,{image_str}"

    image_parts = [
        {"type": "image_url", "image_url": {"url": normalize_image_data(img)}}
        for img in request.images
    ]

    model = get_chat_model(
        model="qwen3-vl-plus",
        response_format={"type": "json_object"},
    )
    messages = [
        SystemMessage(content=DRUG_PARSER_PROMPT),
        HumanMessage(content=image_parts),
    ]
    result = model.invoke(messages)

    try:
        data = json.loads(result.content)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="模型返回非 JSON 内容") from exc

    return ApiResponse.success(data=data, message="解析成功")
