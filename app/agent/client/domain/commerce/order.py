"""
客户端 commerce 订单工具。
"""

from __future__ import annotations

from langchain_core.tools import tool

from app.agent.client.domain.commerce.order_schema import (
    OpenUserOrderListRequest,
    OrderNoRequest,
)
from app.core.agent.agent_event_bus import enqueue_final_sse_response
from app.core.agent.tool_cache import CLIENT_COMMERCE_TOOL_CACHE_PROFILE, tool_cacheable
from app.schemas.http_response import HttpResponse
from app.schemas.sse_response import (
    Action,
    AssistantResponse,
    Content,
    MessageType,
    OrderStatusValue,
    UserOrderListPayload,
)
from app.utils.http_client import HttpClient

# 前端动作用统一优先级下发，确保页面跳转类动作优先被消费。
_DEFAULT_ACTION_PRIORITY = 100
# 订单状态码到用户文案的映射，用于生成自然语言反馈。
_ORDER_STATUS_LABELS: dict[str, str] = {
    "PENDING_PAYMENT": "待支付",
    "PENDING_SHIPMENT": "待发货",
    "PENDING_RECEIPT": "待收货",
    "COMPLETED": "已完成",
    "CANCELLED": "已取消",
}


def _build_order_list_message(order_status: OrderStatusValue | None) -> str:
    """
    功能描述：
        根据订单状态生成给用户看的打开页面提示语。

    参数说明：
        order_status (OrderStatusValue | None): 订单状态筛选值。

    返回值：
        str: 打开订单列表的确认文案。

    异常说明：
        无。
    """

    if order_status is None:
        return "已为你打开订单列表"
    status_label = _ORDER_STATUS_LABELS.get(order_status, order_status)
    return f"已为你打开{status_label}订单列表"


def _build_order_list_action_response(
        order_status: OrderStatusValue | None,
) -> AssistantResponse:
    """
    功能描述：
        构建“打开订单列表”动作响应。

    参数说明：
        order_status (OrderStatusValue | None): 订单状态筛选值。

    返回值：
        AssistantResponse: 打开订单列表的动作响应。

    异常说明：
        无。
    """

    message = _build_order_list_message(order_status)
    return AssistantResponse(
        type=MessageType.ACTION,
        content=Content(message=message),
        action=Action(
            type="navigate",
            target="user_order_list",
            payload=UserOrderListPayload(orderStatus=order_status),
            priority=_DEFAULT_ACTION_PRIORITY,
        ),
    )


@tool(
    args_schema=OpenUserOrderListRequest,
    description=(
            "打开用户订单列表页面。"
            "调用时机：用户明确要求打开、进入或查看订单列表时。"
            "如果用户提到了待支付、待发货、待收货、已完成、已取消等状态，"
            "请传入对应的 orderStatus；否则不要传。"
    ),
)
async def open_user_order_list(orderStatus: OrderStatusValue | None = None) -> str:
    """
    功能描述：
        将“打开订单列表”动作加入当前请求的最终 SSE 响应队列。

    参数说明：
        orderStatus (OrderStatusValue | None): 订单状态筛选值。

    返回值：
        str: 给用户看的打开订单列表提示语。

    异常说明：
        无。
    """

    response = _build_order_list_action_response(orderStatus)
    enqueue_final_sse_response(response)
    return _build_order_list_message(orderStatus)


@tool(
    args_schema=OrderNoRequest,
    description=(
            "获取订单详情。"
            "调用时机：用户已经给出订单编号，想查看金额、商品、收货信息、支付状态或物流摘要时。"
    ),
)
@tool_cacheable(
    CLIENT_COMMERCE_TOOL_CACHE_PROFILE,
    tool_name="get_order_detail",
)
async def get_order_detail(order_no: str) -> dict:
    """
    功能描述：
        获取客户端订单详情。

    参数说明：
        order_no (str): 订单编号。

    返回值：
        dict: 订单详情数据。

    异常说明：
        无；底层 HTTP 异常由上层抛出。
    """

    async with HttpClient() as client:
        response = await client.get(
            url=f"/agent/client/order/{order_no}",
        )
        return HttpResponse.parse_data(response)


@tool(
    args_schema=OrderNoRequest,
    description=(
            "获取订单物流。"
            "调用时机：用户已经给出订单编号，想查看是否发货、物流公司、运单号或物流轨迹时。"
    ),
)
@tool_cacheable(
    CLIENT_COMMERCE_TOOL_CACHE_PROFILE,
    tool_name="get_order_shipping",
)
async def get_order_shipping(order_no: str) -> dict:
    """
    功能描述：
        获取客户端订单物流。

    参数说明：
        order_no (str): 订单编号。

    返回值：
        dict: 订单物流数据。

    异常说明：
        无；底层 HTTP 异常由上层抛出。
    """

    async with HttpClient() as client:
        response = await client.get(
            url=f"/agent/client/order/shipping/{order_no}",
        )
        return HttpResponse.parse_data(response)


@tool(
    args_schema=OrderNoRequest,
    description=(
            "获取订单时间线。"
            "调用时机：用户已经给出订单编号，想查看订单从创建到当前状态的关键过程节点时。"
    ),
)
@tool_cacheable(
    CLIENT_COMMERCE_TOOL_CACHE_PROFILE,
    tool_name="get_order_timeline",
)
async def get_order_timeline(order_no: str) -> dict:
    """
    功能描述：
        获取客户端订单时间线。

    参数说明：
        order_no (str): 订单编号。

    返回值：
        dict: 订单时间线数据。

    异常说明：
        无；底层 HTTP 异常由上层抛出。
    """

    async with HttpClient() as client:
        response = await client.get(
            url=f"/agent/client/order/timeline/{order_no}",
        )
        return HttpResponse.parse_data(response)


@tool(
    args_schema=OrderNoRequest,
    description=(
            "校验是否可取消订单。"
            "调用时机：用户已经给出订单编号，想确认当前订单能不能取消以及原因时。"
    ),
)
@tool_cacheable(
    CLIENT_COMMERCE_TOOL_CACHE_PROFILE,
    tool_name="check_order_cancelable",
)
async def check_order_cancelable(order_no: str) -> dict:
    """
    功能描述：
        校验客户端订单是否可取消。

    参数说明：
        order_no (str): 订单编号。

    返回值：
        dict: 订单取消资格校验结果。

    异常说明：
        无；底层 HTTP 异常由上层抛出。
    """

    async with HttpClient() as client:
        response = await client.get(
            url=f"/agent/client/order/cancel-check/{order_no}",
        )
        return HttpResponse.parse_data(response)


__all__ = [
    "check_order_cancelable",
    "get_order_detail",
    "get_order_shipping",
    "get_order_timeline",
    "open_user_order_list",
]
