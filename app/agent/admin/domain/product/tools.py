from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool

from app.agent.admin.domain.tools import _normalize_id_list, format_ids_to_string
from app.agent.admin.domain.product.schema import (
    DrugDetailRequest,
    MallProductListQueryRequest,
    ProductInfoRequest,
)
from app.core.agent.agent_tool_events import tool_call_status
from app.schemas.http_response import HttpResponse
from app.utils.http_client import HttpClient


@tool(
    args_schema=MallProductListQueryRequest,
    description=(
            "查看商城商品列表，如果不传入任何参数，只传递分页信息的话，这边默认返回最新的前N条数据，"
            "支持按名称、价格区间、分类等条件筛选。"
            "调用时机：当用户关注于商城内的商品信息时。"
    ),
)
@tool_call_status(
    tool_name="获取商品列表",
    start_message="正在查询商品列表",
    error_message="获取商品列表失败",
    timely_message="商品列表正在持续处理中",
)
async def get_product_list(
        page_num: int = 1,
        page_size: int = 10,
        id: Optional[int] = None,
        name: Optional[str] = None,
        category_id: Optional[int] = None,
        status: Optional[int] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
) -> dict:
    """搜索商城商品列表。"""

    async with HttpClient() as client:
        params = {
            "pageNum": page_num,
            "pageSize": page_size,
            "id": id,
            "name": name,
            "categoryId": category_id,
            "status": status,
            "minPrice": min_price,
            "maxPrice": max_price,
        }
        response = await client.get(url="/agent/admin/product/list", params=params)
        return HttpResponse.parse_data(response)


@tool(
    args_schema=ProductInfoRequest,
    description=(
            "根据商品ID获取详细信息，支持批量查询。"
            "参数传递规则：product_id 必须是字符串数组 List[str]，例如 "
            "{\"product_id\": [\"2001\", \"2003\"]}；"
            "不要传逗号拼接字符串。"
            "调用时机：用户明确询问某个或某些商品的详细信息时。"
    ),
)
@tool_call_status(
    tool_name="获取商品详情",
    start_message="正在查询商品详情",
    error_message="获取商品详情失败",
    timely_message="商品详情正在持续处理中",
)
async def get_product_detail(product_id: list[str]) -> dict:
    """根据商品ID获取详细信息，支持批量查询。"""

    normalized_ids = _normalize_id_list(product_id, field_name="product_id")
    ids_str = format_ids_to_string(normalized_ids)
    async with HttpClient() as client:
        response = await client.get(url=f"/agent/admin/product/{ids_str}")
        return HttpResponse.parse_data(response)


@tool(
    args_schema=DrugDetailRequest,
    description=(
            "根据商品ID获取药品详细信息，包括说明书、适应症、用法用量等，支持批量查询。"
            "参数传递规则：product_id 必须是字符串数组 List[str]，例如 "
            "{\"product_id\": [\"2001\", \"2003\"]}；"
            "不要传逗号拼接字符串。"
            "调用时机：用户询问药品说明书、适应症、用法用量等信息时。"
    ),
)
@tool_call_status(
    tool_name="获取药品详情",
    start_message="正在查询药品详情",
    error_message="获取药品详情失败",
    timely_message="药品详情正在持续处理中",
)
async def get_drug_detail(product_id: list[str]) -> dict:
    """根据药品商品ID获取详细药品信息。"""

    normalized_ids = _normalize_id_list(product_id, field_name="product_id")
    ids_str = format_ids_to_string(normalized_ids)
    async with HttpClient() as client:
        response = await client.get(url=f"/agent/admin/product/drug/{ids_str}")
        return HttpResponse.parse_data(response)
