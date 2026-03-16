from __future__ import annotations

import uuid
from typing import Annotated

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field

from app.core.agent.agent_event_bus import enqueue_final_sse_response
from app.schemas.sse_response import AssistantResponse, MessageType, ProductCard, ProductCardData

ProductIdValue = Annotated[int, Field(ge=1, description="商品 ID，必须为正整数")]


class SendProductCardRequest(BaseModel):
    """发送商品卡片工具参数。"""

    model_config = ConfigDict(extra="forbid")

    productIds: list[ProductIdValue] = Field(
        min_length=1,
        description="商品 ID 列表，仅传需要前端渲染的商品 ID。",
    )


def _build_product_card_response(product_ids: list[int]) -> AssistantResponse:
    return AssistantResponse(
        type=MessageType.CARD,
        card=ProductCard(
            data=ProductCardData(productIds=list(product_ids)),
        ),
        meta={
            "card_uuid": str(uuid.uuid4()),
        },
    )


@tool(
    args_schema=SendProductCardRequest,
    description=(
            "向前端发送商品卡片。"
            "调用时机：已经筛出要推荐的商品后，希望在本轮回答文本结束后展示商品卡片时"
            "这边传递获得的商品 ID 即可"
    ),
)
async def send_product_card(productIds: list[int]) -> str:
    """将商品卡片加入当前请求的最终 SSE 响应队列。"""

    enqueue_final_sse_response(_build_product_card_response(productIds))
    return "已准备商品卡片"


__all__ = [
    "SendProductCardRequest",
    "send_product_card",
]
