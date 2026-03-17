from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.sse_response import Card
from app.utils.http_client import HttpClient

_DEFAULT_PRODUCT_CARD_TITLE = "为您推荐以下商品"


class PurchaseCardFieldMeta(BaseModel):
    """商品购买卡片接口返回的字段说明元数据。"""

    model_config = ConfigDict(extra="ignore")

    entityDescription: str | None = Field(default=None, description="实体说明")
    fieldDescriptions: dict[str, str] = Field(
        default_factory=dict,
        description="字段语义说明",
    )


class PurchaseCardItem(BaseModel):
    """商品购买卡片接口中的单个商品项。"""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., description="商品 ID")
    name: str = Field(..., description="商品名称")
    image: str = Field(..., description="商品主图")
    price: str = Field(..., description="商品销售价")
    spec: str | None = Field(default=None, description="商品规格")
    efficacy: str | None = Field(default=None, description="功效/适应症")
    prescription: bool | None = Field(default=None, description="是否处方药")
    stock: int | None = Field(default=None, description="库存")


class PurchaseCardResponseData(BaseModel):
    """商品购买卡片接口业务数据。"""

    model_config = ConfigDict(extra="ignore")

    totalPrice: str = Field(default="0.00", description="商品总价")
    items: list[PurchaseCardItem] = Field(
        default_factory=list,
        description="商品卡片列表",
    )
    meta: PurchaseCardFieldMeta | None = Field(
        default=None,
        description="字段语义元数据",
    )


class ProductCardProduct(BaseModel):
    """前端商品卡片中的单个商品项。"""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="商品 ID")
    name: str = Field(..., description="商品名称")
    image: str = Field(..., description="商品主图")
    price: str = Field(..., description="商品销售价")


class ProductCardData(BaseModel):
    """前端商品卡片数据。"""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        default=_DEFAULT_PRODUCT_CARD_TITLE,
        description="商品卡片标题",
    )
    products: list[ProductCardProduct] = Field(
        default_factory=list,
        description="商品展示列表",
    )


def _build_purchase_card_url(product_ids: list[int]) -> str:
    normalized_ids = ",".join(str(product_id) for product_id in product_ids)
    return f"/agent/client/purchase_cards/{normalized_ids}"


def _map_product_items(
        *,
        product_ids: list[int],
        items: list[PurchaseCardItem],
) -> list[ProductCardProduct]:
    items_by_id: dict[str, PurchaseCardItem] = {}
    for item in items:
        item_id = item.id.strip()
        if item_id and item_id not in items_by_id:
            items_by_id[item_id] = item

    products: list[ProductCardProduct] = []
    for product_id in product_ids:
        item = items_by_id.get(str(product_id))
        if item is None:
            continue
        products.append(
            ProductCardProduct(
                id=item.id,
                name=item.name,
                image=item.image,
                price=item.price,
            )
        )
    return products


async def render_product_card(product_ids: list[int]) -> Card | None:
    """请求业务端补全商品卡片；无有效商品时返回 `None`。"""

    if not product_ids:
        return None

    async with HttpClient() as client:
        payload: Any = await client.get(
            _build_purchase_card_url(product_ids),
            response_format="json",
        )

    response_data = PurchaseCardResponseData.model_validate(payload)
    products = _map_product_items(
        product_ids=product_ids,
        items=response_data.items,
    )
    if not products:
        return None

    card_data = ProductCardData(products=products)
    return Card(
        type="product-card",
        data=card_data.model_dump(mode="json"),
    )


__all__ = [
    "ProductCardData",
    "ProductCardProduct",
    "PurchaseCardFieldMeta",
    "PurchaseCardItem",
    "PurchaseCardResponseData",
    "render_product_card",
]
