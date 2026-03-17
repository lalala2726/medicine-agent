from __future__ import annotations

import uuid
from typing import Annotated

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field

from app.agent.services.card_render_service import render_product_card
from app.core.agent.agent_event_bus import enqueue_final_sse_response
from app.schemas.sse_response import AssistantResponse, Card, MessageType

ProductIdValue = Annotated[int, Field(ge=1, description="商品 ID，必须为正整数")]


class SendProductCardRequest(BaseModel):
    """发送商品卡片工具参数。"""

    model_config = ConfigDict(extra="forbid")

    productIds: list[ProductIdValue] = Field(
        min_length=1,
        description="商品 ID 列表，仅传需要前端渲染的商品 ID。",
    )


def _build_product_card_response(card: Card) -> AssistantResponse:
    return AssistantResponse(
        type=MessageType.CARD,
        card=card,
        meta={
            "card_uuid": str(uuid.uuid4()),
        },
    )


@tool(
    args_schema=SendProductCardRequest,
    description=(
            "向前端发送商品卡片。"
            "调用时机：已经筛出要推荐的商品后，希望在本轮回答文本结束后展示商品卡片时。"
            "这边仅需传递商品 ID 列表，工具会自动请求业务端获取商品信息并渲染成卡片。"
    ),
)
async def send_product_card(productIds: list[int]) -> str:
    """将商品卡片加入当前请求的最终 SSE 响应队列。"""

    card = await render_product_card(productIds)
    if card is None:
        return "__ERROR__:未从业务端获取到商品信息无法发送商品卡片"
    enqueue_final_sse_response(_build_product_card_response(card))
    return "__SUCCESS__"


__all__ = [
    "SendProductCardRequest",
    "send_product_card",
]
