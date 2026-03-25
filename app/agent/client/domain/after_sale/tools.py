from __future__ import annotations

from langchain_core.tools import tool

from app.agent.client.domain.after_sale.schema import (
    AfterSaleEligibilityRequest,
    AfterSaleNoRequest,
)
from app.core.agent.tool_cache import CLIENT_COMMERCE_TOOL_CACHE_PROFILE, tool_cacheable
from app.schemas.http_response import HttpResponse
from app.utils.http_client import HttpClient


@tool(
    args_schema=AfterSaleNoRequest,
    description=(
            "获取售后详情。"
            "调用时机：用户已提供售后单号，想查看售后状态、退款金额、驳回原因或处理时间线时。"
    ),
)
@tool_cacheable(
    CLIENT_COMMERCE_TOOL_CACHE_PROFILE,
    tool_name="get_after_sale_detail",
)
async def get_after_sale_detail(after_sale_no: str) -> dict:
    """获取客户端售后详情。"""

    async with HttpClient() as client:
        response = await client.get(
            url=f"/agent/client/after-sale/{after_sale_no}",
        )
        return HttpResponse.parse_data(response)


@tool(
    args_schema=AfterSaleEligibilityRequest,
    description=(
            "校验售后资格。"
            "调用时机：用户想确认某笔订单或订单项当前能不能申请退款、退货或换货时。"
    ),
)
@tool_cacheable(
    CLIENT_COMMERCE_TOOL_CACHE_PROFILE,
    tool_name="check_after_sale_eligibility",
)
async def check_after_sale_eligibility(
        order_no: str,
        order_item_id: int | None = None,
) -> dict:
    """校验客户端售后资格。"""

    async with HttpClient() as client:
        response = await client.get(
            url="/agent/client/after-sale/eligibility",
            params={
                "orderNo": order_no,
                "orderItemId": order_item_id,
            },
        )
        return HttpResponse.parse_data(response)


__all__ = [
    "check_after_sale_eligibility",
    "get_after_sale_detail",
]
