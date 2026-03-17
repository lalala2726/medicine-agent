from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.sse_response import Card
from app.utils.http_client import HttpClient

_MONEY_PRECISION = Decimal("0.00")
_DEFAULT_PRODUCT_CARD_TITLE = "为您推荐以下商品"
_DEFAULT_PRODUCT_PURCHASE_CARD_TITLE = "请确认要购买的商品"


class PurchaseCardFieldMeta(BaseModel):
    """商品卡片接口返回的字段说明元数据。"""

    model_config = ConfigDict(extra="ignore")

    entityDescription: str | None = Field(default=None, description="实体说明")
    fieldDescriptions: dict[str, str] = Field(
        default_factory=dict,
        description="字段语义说明",
    )


class BaseProductCardItem(BaseModel):
    """商品卡片上游接口中的公共商品字段。"""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., description="商品 ID")
    name: str = Field(..., description="商品名称")
    image: str = Field(..., description="商品主图")
    price: Decimal = Field(default=Decimal("0.00"), description="商品销售价")


class ProductCardItem(BaseProductCardItem):
    """推荐商品卡接口中的单个商品项。"""

    model_config = ConfigDict(extra="ignore")

    spec: str | None = Field(default=None, description="商品规格")
    efficacy: str | None = Field(default=None, description="功效/适应症")
    prescription: bool | None = Field(default=None, description="是否处方药")
    stock: int | None = Field(default=None, description="库存")


class ProductCardResponseData(BaseModel):
    """推荐商品卡接口业务数据。"""

    model_config = ConfigDict(extra="ignore")

    totalPrice: Decimal = Field(default=Decimal("0.00"), description="商品总价")
    items: list[ProductCardItem] = Field(
        default_factory=list,
        description="商品卡片列表",
    )
    meta: PurchaseCardFieldMeta | None = Field(
        default=None,
        description="字段语义元数据",
    )


class ProductCardProduct(BaseModel):
    """前端推荐商品卡片中的单个商品项。"""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="商品 ID")
    name: str = Field(..., description="商品名称")
    image: str = Field(..., description="商品主图")
    price: str = Field(..., description="商品销售价")


class ProductCardData(BaseModel):
    """前端推荐商品卡片数据。"""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        default=_DEFAULT_PRODUCT_CARD_TITLE,
        description="商品卡片标题",
    )
    products: list[ProductCardProduct] = Field(
        default_factory=list,
        description="商品展示列表",
    )


class ProductPurchaseCardRequestItem(BaseModel):
    """商品购买卡片请求中的单个购买项。"""

    model_config = ConfigDict(extra="forbid")

    productId: int = Field(..., gt=0, description="商品 ID，必须大于 0")
    quantity: int = Field(..., gt=0, description="购买数量，必须大于 0")


class ProductPurchaseCardItem(BaseProductCardItem):
    """商品购买卡片接口中的单个商品项。"""

    model_config = ConfigDict(extra="ignore")

    quantity: int = Field(..., gt=0, description="购买数量")
    spec: str | None = Field(default=None, description="商品规格")
    efficacy: str | None = Field(default=None, description="功效/适应症")
    prescription: bool | None = Field(default=None, description="是否处方药")
    stock: int | None = Field(default=None, description="库存")


class ProductPurchaseCardResponseData(BaseModel):
    """商品购买卡片接口业务数据。"""

    model_config = ConfigDict(extra="ignore")

    totalPrice: Decimal = Field(default=Decimal("0.00"), description="商品总价")
    items: list[ProductPurchaseCardItem] = Field(
        default_factory=list,
        description="商品购买卡片列表",
    )
    meta: PurchaseCardFieldMeta | None = Field(
        default=None,
        description="字段语义元数据",
    )


class ProductPurchaseCardProduct(BaseModel):
    """前端商品购买卡片中的单个商品项。"""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="商品 ID")
    name: str = Field(..., description="商品名称")
    image: str = Field(..., description="商品主图")
    price: str = Field(..., description="商品销售价")
    quantity: int = Field(..., gt=0, description="购买数量")


class ProductPurchaseCardData(BaseModel):
    """前端商品购买卡片数据。"""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        default=_DEFAULT_PRODUCT_PURCHASE_CARD_TITLE,
        description="商品购买卡片标题",
    )
    products: list[ProductPurchaseCardProduct] = Field(
        default_factory=list,
        description="商品购买展示列表",
    )
    total_price: str = Field(default="0.00", description="当前展示商品总价")


def _format_money(value: Decimal | int | float | str) -> str:
    """将业务端价格统一格式化为两位小数字符串。"""

    try:
        normalized_value = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        normalized_value = Decimal("0")
    return str(normalized_value.quantize(_MONEY_PRECISION, rounding=ROUND_HALF_UP))


def _build_product_card_url(product_ids: list[int]) -> str:
    """构建推荐商品卡片补全接口地址。"""

    normalized_ids = ",".join(str(product_id) for product_id in product_ids)
    return f"/agent/client/card/product/{normalized_ids}"


def _map_product_items(
        *,
        product_ids: list[int],
        items: list[ProductCardItem],
) -> list[ProductCardProduct]:
    """按传入商品 ID 顺序映射前端推荐商品卡片。"""

    items_by_id: dict[str, ProductCardItem] = {}
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
                price=_format_money(item.price),
            )
        )
    return products


def _normalize_purchase_card_request_items(
        items: list[ProductPurchaseCardRequestItem | dict[str, Any]],
) -> list[ProductPurchaseCardRequestItem]:
    """统一校验并规范化商品购买卡片请求项。"""

    return [
        item
        if isinstance(item, ProductPurchaseCardRequestItem)
        else ProductPurchaseCardRequestItem.model_validate(item)
        for item in items
    ]


def _map_product_purchase_items(
        *,
        request_items: list[ProductPurchaseCardRequestItem],
        items: list[ProductPurchaseCardItem],
) -> tuple[list[ProductPurchaseCardProduct], str]:
    """按请求顺序映射前端商品购买卡片，并计算展示总价。"""

    items_by_id: dict[str, ProductPurchaseCardItem] = {}
    for item in items:
        item_id = item.id.strip()
        if item_id and item_id not in items_by_id:
            items_by_id[item_id] = item

    products: list[ProductPurchaseCardProduct] = []
    total_price = Decimal("0")
    for request_item in request_items:
        item = items_by_id.get(str(request_item.productId))
        if item is None:
            continue
        products.append(
            ProductPurchaseCardProduct(
                id=item.id,
                name=item.name,
                image=item.image,
                price=_format_money(item.price),
                quantity=request_item.quantity,
            )
        )
        total_price += item.price * request_item.quantity

    return products, _format_money(total_price)


async def render_product_card(product_ids: list[int]) -> Card | None:
    """请求业务端补全推荐商品卡片；无有效商品时返回 `None`。"""

    if not product_ids:
        return None

    async with HttpClient() as client:
        payload: Any = await client.get(
            _build_product_card_url(product_ids),
            response_format="json",
        )

    response_data = ProductCardResponseData.model_validate(payload)
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


async def render_product_purchase_card(
        items: list[ProductPurchaseCardRequestItem | dict[str, Any]],
) -> Card | None:
    """请求业务端补全商品购买卡片；无有效商品时返回 `None`。"""

    request_items = _normalize_purchase_card_request_items(items)
    if not request_items:
        return None

    async with HttpClient() as client:
        payload: Any = await client.post(
            "/agent/client/card/purchase_cards",
            json={
                "items": [
                    item.model_dump(mode="json")
                    for item in request_items
                ]
            },
            response_format="json",
        )

    response_data = ProductPurchaseCardResponseData.model_validate(payload)
    products, total_price = _map_product_purchase_items(
        request_items=request_items,
        items=response_data.items,
    )
    if not products:
        return None

    card_data = ProductPurchaseCardData(
        products=products,
        total_price=total_price,
    )
    return Card(
        type="product-purchase-card",
        data=card_data.model_dump(mode="json"),
    )


__all__ = [
    "ProductCardData",
    "ProductCardItem",
    "ProductCardProduct",
    "ProductCardResponseData",
    "ProductPurchaseCardData",
    "ProductPurchaseCardItem",
    "ProductPurchaseCardProduct",
    "ProductPurchaseCardRequestItem",
    "ProductPurchaseCardResponseData",
    "PurchaseCardFieldMeta",
    "render_product_card",
    "render_product_purchase_card",
]
