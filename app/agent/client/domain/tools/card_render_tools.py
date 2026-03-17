from __future__ import annotations

import uuid
from typing import Annotated

from langchain_core.tools import tool
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from app.agent.services.card_render_schema import ProductPurchaseCardRequestItem
from app.agent.services.card_render_service import (
    render_product_card,
    render_product_purchase_card,
)
from app.core.agent.agent_event_bus import enqueue_final_sse_response
from app.schemas.sse_response import AssistantResponse, Card, MessageType

ProductIdValue = Annotated[int, Field(ge=1, description="商品 ID，必须为正整数")]


class SendProductCardRequest(BaseModel):
    """发送商品卡片工具参数。"""

    model_config = ConfigDict(extra="forbid")

    productIds: list[ProductIdValue] = Field(
        min_length=1,
        description="需要展示为推荐商品卡片的商品 ID 列表。",
    )


class SendProductPurchaseCardItem(ProductPurchaseCardRequestItem):
    """发送商品购买卡片时的单个商品项参数。"""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "productId": 101,
                "quantity": 2,
            }
        },
    )

    productId: int = Field(
        ...,
        gt=0,
        description="待购买商品的商品 ID，必填，且必须大于 0。",
    )
    quantity: int = Field(
        ...,
        gt=0,
        description="该商品的购买数量，必填，且必须大于 0。",
    )


class SendProductPurchaseCardRequest(BaseModel):
    """发送商品购买卡片工具参数。"""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "items": [
                    {
                        "productId": 101,
                        "quantity": 2,
                    },
                    {
                        "productId": 205,
                        "quantity": 1,
                    },
                ]
            }
        },
    )

    items: list[SendProductPurchaseCardItem] = Field(
        min_length=1,
        description="商品购买项列表，每个元素表示一个待确认购买的商品及其数量。",
    )


def _build_card_response(card: Card) -> AssistantResponse:
    """构建通用卡片 SSE 响应。"""

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
    ),
)
async def send_product_card(productIds: list[int]) -> str:
    """将商品卡片加入当前请求的最终 SSE 响应队列。"""

    try:
        card = await render_product_card(productIds)
    except Exception as exc:
        logger.opt(exception=exc).error(
            "send_product_card failed product_ids={}",
            productIds,
        )
        raise

    if card is None:
        logger.warning(
            "send_product_card skipped because card is empty product_ids={}",
            productIds,
        )
        return "__ERROR__:未从业务端获取到商品信息无法发送商品卡片"

    enqueue_final_sse_response(_build_card_response(card))
    logger.info(
        "send_product_card succeeded product_ids={} card_type={}",
        productIds,
        card.type,
    )
    return "__SUCCESS__"


@tool(
    args_schema=SendProductPurchaseCardRequest,
    description=(
            "向前端发送商品购买卡片。"
            "调用时机：已经明确用户准备购买哪些商品以及对应数量后，希望在本轮回答文本结束后展示购买确认卡片时。"
    ),
)
async def send_product_purchase_card(
        items: list[SendProductPurchaseCardItem],
) -> str:
    """将商品购买卡片加入当前请求的最终 SSE 响应队列。"""

    print("调用了")

    try:
        card = await render_product_purchase_card(items)
    except Exception as exc:
        logger.opt(exception=exc).error(
            "send_product_purchase_card failed items={}",
            [item.model_dump(mode="json") for item in items],
        )
        raise

    if card is None:
        logger.warning(
            "send_product_purchase_card skipped because card is empty items={}",
            [item.model_dump(mode="json") for item in items],
        )
        return "__ERROR__:未从业务端获取到商品购买信息无法发送商品购买卡片"

    enqueue_final_sse_response(_build_card_response(card))
    logger.info(
        "send_product_purchase_card succeeded item_count={} card_type={}",
        len(items),
        card.type,
    )
    return "__SUCCESS__"


__all__ = [
    "SendProductCardRequest",
    "SendProductPurchaseCardRequest",
    "send_product_card",
    "send_product_purchase_card",
]
