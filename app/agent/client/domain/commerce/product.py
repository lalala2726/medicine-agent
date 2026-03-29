"""
客户端 commerce 商品工具。
"""

from __future__ import annotations

from langchain_core.tools import tool

from app.agent.client.domain.commerce.product_schema import (
    ProductIdRequest,
    ProductSearchRequest,
)
from app.core.agent.tool_cache import CLIENT_COMMERCE_TOOL_CACHE_PROFILE, tool_cacheable
from app.schemas.http_response import HttpResponse
from app.utils.http_client import HttpClient


@tool(
    args_schema=ProductSearchRequest,
    description=(
            "搜索商品。"
            "调用时机：用户想找某类商品、按用途挑选商品、按关键词搜索商品时。"
    ),
)
@tool_cacheable(
    CLIENT_COMMERCE_TOOL_CACHE_PROFILE,
    tool_name="search_products",
)
async def search_products(
        keyword: str | None = None,
        category_name: str | None = None,
        usage: str | None = None,
        page_num: int = 1,
        page_size: int = 10,
) -> dict:
    """
    功能描述：
        搜索客户端商品列表。

    参数说明：
        keyword (str | None): 商品关键词。
        category_name (str | None): 商品分类名称。
        usage (str | None): 商品用途或适用场景。
        page_num (int): 页码。
        page_size (int): 每页数量。

    返回值：
        dict: 商品搜索结果分页数据。

    异常说明：
        无；底层 HTTP 异常由上层抛出。
    """

    async with HttpClient() as client:
        response = await client.get(
            url="/agent/client/product/search",
            params={
                "keyword": keyword,
                "categoryName": category_name,
                "usage": usage,
                "pageNum": page_num,
                "pageSize": page_size,
            },
        )
        return HttpResponse.parse_data(response)


@tool(
    args_schema=ProductIdRequest,
    description=(
            "获取商品详情。"
            "调用时机：用户明确询问某个商品的价格、库存、分类、图片或药品说明信息时。"
    ),
)
@tool_cacheable(
    CLIENT_COMMERCE_TOOL_CACHE_PROFILE,
    tool_name="get_product_detail",
)
async def get_product_detail(product_id: int) -> dict:
    """
    功能描述：
        获取客户端商品详情。

    参数说明：
        product_id (int): 商品 ID。

    返回值：
        dict: 商品详情数据。

    异常说明：
        无；底层 HTTP 异常由上层抛出。
    """

    async with HttpClient() as client:
        response = await client.get(
            url=f"/agent/client/product/{product_id}",
        )
        return HttpResponse.parse_data(response)


@tool(
    args_schema=ProductIdRequest,
    description=(
            "获取商品规格属性。"
            "调用时机：用户询问商品成分、包装、有效期、注意事项、禁忌、说明书等更细的属性时。"
    ),
)
@tool_cacheable(
    CLIENT_COMMERCE_TOOL_CACHE_PROFILE,
    tool_name="get_product_spec",
)
async def get_product_spec(product_id: int) -> dict:
    """
    功能描述：
        获取客户端商品规格属性。

    参数说明：
        product_id (int): 商品 ID。

    返回值：
        dict: 商品规格属性数据。

    异常说明：
        无；底层 HTTP 异常由上层抛出。
    """

    async with HttpClient() as client:
        response = await client.get(
            url=f"/agent/client/product/spec/{product_id}",
        )
        return HttpResponse.parse_data(response)


__all__ = [
    "get_product_detail",
    "get_product_spec",
    "search_products",
]
