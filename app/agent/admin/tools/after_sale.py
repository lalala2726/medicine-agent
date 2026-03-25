"""
售后领域工具。
"""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.core.agent.agent_tool_events import tool_call_status
from app.schemas.http_response import HttpResponse
from app.utils.http_client import HttpClient


class AfterSaleListRequest(BaseModel):
    """
    功能描述：
        售后列表查询入参模型。

    参数说明：
        page_num (int): 页码。
        page_size (int): 每页数量。
        after_sale_type (str | None): 售后类型。
        after_sale_status (str | None): 售后状态。
        order_no (str | None): 订单号。
        user_id (int | None): 用户 ID。
        apply_reason (str | None): 申请原因。

    返回值：
        无（数据模型定义）。

    异常说明：
        无。
    """

    page_num: int = Field(default=1, ge=1, description="页码，从 1 开始")
    page_size: int = Field(default=10, ge=1, le=200, description="每页数量，范围 1-200")
    after_sale_type: Optional[str] = Field(default=None, description="售后类型，例如 REFUND_ONLY")
    after_sale_status: Optional[str] = Field(default=None, description="售后状态，例如 PENDING、APPROVED")
    order_no: Optional[str] = Field(default=None, description="订单编号，精确匹配")
    user_id: Optional[int] = Field(default=None, ge=1, description="用户 ID")
    apply_reason: Optional[str] = Field(default=None, description="申请原因，例如 DAMAGED")


class AfterSaleIdRequest(BaseModel):
    """
    功能描述：
        售后详情查询入参模型。

    参数说明：
        after_sale_id (int): 售后申请 ID。

    返回值：
        无（数据模型定义）。

    异常说明：
        无。
    """

    after_sale_id: int = Field(ge=1, description="售后申请 ID")


@tool(
    args_schema=AfterSaleListRequest,
    description=(
            "分页查询售后申请列表，支持按售后类型、状态、订单号、用户 ID 和申请原因筛选。"
            "适用于定位售后单范围和查看待处理售后。"
    ),
)
@tool_call_status(
    tool_name="查询售后列表",
    start_message="正在查询售后列表",
    error_message="查询售后列表失败",
    timely_message="售后列表正在持续处理中",
)
async def after_sale_list(
        page_num: int = 1,
        page_size: int = 10,
        after_sale_type: Optional[str] = None,
        after_sale_status: Optional[str] = None,
        order_no: Optional[str] = None,
        user_id: Optional[int] = None,
        apply_reason: Optional[str] = None,
) -> dict:
    """
    功能描述：
        查询售后申请列表。

    参数说明：
        page_num (int): 页码。
        page_size (int): 每页数量。
        after_sale_type (Optional[str]): 售后类型。
        after_sale_status (Optional[str]): 售后状态。
        order_no (Optional[str]): 订单号。
        user_id (Optional[int]): 用户 ID。
        apply_reason (Optional[str]): 申请原因。

    返回值：
        dict: 售后列表数据。

    异常说明：
        无。
    """

    async with HttpClient() as client:
        params = {
            "pageNum": page_num,
            "pageSize": page_size,
            "afterSaleType": after_sale_type,
            "afterSaleStatus": after_sale_status,
            "orderNo": order_no,
            "userId": user_id,
            "applyReason": apply_reason,
        }
        response = await client.get(
            url="/agent/admin/after-sale/list",
            params=params,
        )
        return HttpResponse.parse_data(response)


@tool(
    args_schema=AfterSaleIdRequest,
    description=(
            "根据售后申请 ID 查询售后详情。"
            "适用于查看处理进度、原因、处理结果和关联订单信息。"
    ),
)
@tool_call_status(
    tool_name="查询售后详情",
    start_message="正在查询售后详情",
    error_message="查询售后详情失败",
    timely_message="售后详情正在持续处理中",
)
async def after_sale_detail(after_sale_id: int) -> dict:
    """
    功能描述：
        根据售后申请 ID 查询售后详情。

    参数说明：
        after_sale_id (int): 售后申请 ID。

    返回值：
        dict: 售后详情数据。

    异常说明：
        无。
    """

    async with HttpClient() as client:
        response = await client.get(
            url=f"/agent/admin/after-sale/detail/{after_sale_id}",
        )
        return HttpResponse.parse_data(response)


__all__ = [
    "AfterSaleIdRequest",
    "AfterSaleListRequest",
    "after_sale_detail",
    "after_sale_list",
]
