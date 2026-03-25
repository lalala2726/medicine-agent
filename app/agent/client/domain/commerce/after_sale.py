"""
客户端 commerce 售后工具。
"""

from __future__ import annotations

from langchain_core.tools import tool

from app.agent.client.domain.commerce.after_sale_schema import (
    AfterSaleEligibilityRequest,
    AfterSaleNoRequest,
    OpenUserAfterSaleListRequest,
)
from app.core.agent.agent_event_bus import enqueue_final_sse_response
from app.core.agent.tool_cache import CLIENT_COMMERCE_TOOL_CACHE_PROFILE, tool_cacheable
from app.schemas.http_response import HttpResponse
from app.schemas.sse_response import (
    Action,
    AfterSaleStatusValue,
    AssistantResponse,
    Content,
    MessageType,
    UserAfterSaleListPayload,
)
from app.utils.http_client import HttpClient

# 前端动作用统一优先级下发，确保页面跳转类动作优先被消费。
_DEFAULT_ACTION_PRIORITY = 100
# 售后状态码到用户文案的映射，用于生成自然语言反馈。
_AFTER_SALE_STATUS_LABELS: dict[str, str] = {
    "PENDING": "待审核",
    "APPROVED": "已通过",
    "REJECTED": "已拒绝",
    "PROCESSING": "处理中",
    "COMPLETED": "已完成",
    "CANCELLED": "已取消",
}


def _build_after_sale_list_message(
        after_sale_status: AfterSaleStatusValue | None,
) -> str:
    """
    功能描述：
        根据售后状态生成给用户看的打开页面提示语。

    参数说明：
        after_sale_status (AfterSaleStatusValue | None): 售后状态筛选值。

    返回值：
        str: 打开售后列表的确认文案。

    异常说明：
        无。
    """

    if after_sale_status is None:
        return "已为你打开售后列表"
    status_label = _AFTER_SALE_STATUS_LABELS.get(after_sale_status, after_sale_status)
    return f"已为你打开{status_label}售后列表"


def _build_after_sale_list_action_response(
        after_sale_status: AfterSaleStatusValue | None,
) -> AssistantResponse:
    """
    功能描述：
        构建“打开售后列表”动作响应。

    参数说明：
        after_sale_status (AfterSaleStatusValue | None): 售后状态筛选值。

    返回值：
        AssistantResponse: 打开售后列表的动作响应。

    异常说明：
        无。
    """

    message = _build_after_sale_list_message(after_sale_status)
    return AssistantResponse(
        type=MessageType.ACTION,
        content=Content(message=message),
        action=Action(
            type="navigate",
            target="user_after_sale_list",
            payload=UserAfterSaleListPayload(afterSaleStatus=after_sale_status),
            priority=_DEFAULT_ACTION_PRIORITY,
        ),
    )


@tool(
    args_schema=OpenUserAfterSaleListRequest,
    description=(
            "打开用户售后列表页面。"
            "调用时机：用户明确要求打开、进入或查看售后列表时。"
            "如果用户提到了待审核、已通过、已拒绝、处理中、已完成、已取消等状态，"
            "请传入对应的 afterSaleStatus；否则不要传。"
    ),
)
async def open_user_after_sale_list(
        afterSaleStatus: AfterSaleStatusValue | None = None,
) -> str:
    """
    功能描述：
        将“打开售后列表”动作加入当前请求的最终 SSE 响应队列。

    参数说明：
        afterSaleStatus (AfterSaleStatusValue | None): 售后状态筛选值。

    返回值：
        str: 给用户看的打开售后列表提示语。

    异常说明：
        无。
    """

    response = _build_after_sale_list_action_response(afterSaleStatus)
    enqueue_final_sse_response(response)
    return _build_after_sale_list_message(afterSaleStatus)


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
    """
    功能描述：
        获取客户端售后详情。

    参数说明：
        after_sale_no (str): 售后单号。

    返回值：
        dict: 售后详情数据。

    异常说明：
        无；底层 HTTP 异常由上层抛出。
    """

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
    """
    功能描述：
        校验客户端售后资格。

    参数说明：
        order_no (str): 订单编号。
        order_item_id (int | None): 订单项 ID。

    返回值：
        dict: 售后资格校验结果。

    异常说明：
        无；底层 HTTP 异常由上层抛出。
    """

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
    "open_user_after_sale_list",
]
