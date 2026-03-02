from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool

from app.agent.assistant.tools.schemas.after_sale import (
    AdminAfterSaleIdRequest,
    AdminAfterSaleListQueryRequest,
)
from app.core.agent.agent_tool_events import tool_call_status
from app.schemas.http_response import HttpResponse
from app.utils.http_client import HttpClient


@tool(
    args_schema=AdminAfterSaleListQueryRequest,
    description=(
            "分页查询管理端售后申请列表，支持按售后类型、售后状态、订单号、用户 ID、申请原因筛选。"
            "参数传递规则：使用结构化字段，不要把多个筛选条件拼成单字符串。"
    ),
)
@tool_call_status(
    tool_name="查询售后列表",
    start_message="正在查询售后列表",
    error_message="查询售后列表失败",
    timely_message="售后列表正在持续处理中",
)
async def get_admin_after_sale_list(
        page_num: int = 1,
        page_size: int = 10,
        after_sale_type: Optional[str] = None,
        after_sale_status: Optional[str] = None,
        order_no: Optional[str] = None,
        user_id: Optional[int] = None,
        apply_reason: Optional[str] = None,
) -> dict:
    """分页查询管理端售后申请列表。"""

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
    args_schema=AdminAfterSaleIdRequest,
    description=(
            "根据售后申请 ID 查询售后详情。"
            "调用时机：需要查看某个售后申请的处理进度、原因、处理结果或关联订单信息时。"
    ),
)
@tool_call_status(
    tool_name="查询售后详情",
    start_message="正在查询售后详情",
    error_message="查询售后详情失败",
    timely_message="售后详情正在持续处理中",
)
async def get_admin_after_sale_detail(after_sale_id: int) -> dict:
    """根据售后申请 ID 查询售后详情。"""

    async with HttpClient() as client:
        response = await client.get(
            url=f"/agent/admin/after-sale/detail/{after_sale_id}",
        )
        return HttpResponse.parse_data(response)
