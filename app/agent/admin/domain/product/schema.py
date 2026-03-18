"""
商品工具参数模型。
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class MallProductListQueryRequest(BaseModel):
    """
    商城商品列表查询请求参数。

    传参说明：
    1. 至少传分页参数 `page_num/page_size`；
    2. 其余筛选参数按需传，不需要的字段不要传空字符串；
    3. 推荐示例：
       `{"page_num": 1, "page_size": 10, "name": "感冒", "status": 1}`。
    """

    page_num: Optional[int] = Field(default=1, description="页码，从 1 开始，默认为 1")
    page_size: Optional[int] = Field(default=10, description="每页数量，建议 10-50，默认为 10")
    id: Optional[int] = Field(default=None, description="商品ID，精确匹配单个商品")
    name: Optional[str] = Field(
        default=None,
        description="商品名称关键词，支持模糊搜索，例如 '感冒' 可匹配 '感冒灵颗粒'",
    )
    category_id: Optional[int] = Field(default=None, description="商品分类ID，用于筛选特定分类下的商品")
    status: Optional[int] = Field(default=None, description="商品状态筛选：1 表示上架商品，0 表示下架商品")
    min_price: Optional[float] = Field(default=None, description="最低价格，用于价格区间筛选，单位：元")
    max_price: Optional[float] = Field(default=None, description="最高价格，用于价格区间筛选，单位：元")


class ProductInfoRequest(BaseModel):
    """
    商品详情查询请求参数。

    传参示例：
    `{"product_id": ["2001", "2003"]}`
    """

    product_id: list[str] = Field(
        min_length=1,
        description=(
            "商品ID字符串数组（List[str]），支持批量查询。"
            "必须传 JSON 数组，不能传 '2001,2003' 这种逗号字符串。"
        ),
        examples=[["2001"], ["2001", "2003"]],
    )


class DrugDetailRequest(BaseModel):
    """
    药品详情查询请求参数。

    传参示例：
    `{"product_id": ["2001", "2003"]}`
    """

    product_id: list[str] = Field(
        min_length=1,
        description=(
            "药品商品ID字符串数组（List[str]），支持批量查询。"
            "必须传 JSON 数组，不能传逗号拼接字符串。"
        ),
        examples=[["2001"], ["2001", "2003"]],
    )


__all__ = [
    "DrugDetailRequest",
    "MallProductListQueryRequest",
    "ProductInfoRequest",
]
