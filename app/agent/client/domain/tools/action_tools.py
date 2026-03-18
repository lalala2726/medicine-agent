from __future__ import annotations

from langchain_core.tools import tool

from app.agent.client.domain.tools.schema import (
    OpenUserAfterSaleListRequest,
    OpenUserOrderListRequest,
)
from app.core.agent.agent_event_bus import enqueue_final_sse_response
from app.schemas.sse_response import (
    Action,
    AfterSaleStatusValue,
    AssistantResponse,
    Content,
    MessageType,
    OrderStatusValue,
    UserAfterSaleListPayload,
    UserOrderListPayload,
)

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
# 售后状态码到用户文案的映射，用于生成自然语言反馈。
_AFTER_SALE_STATUS_LABELS: dict[str, str] = {
    "PENDING": "待审核",
    "APPROVED": "已通过",
    "REJECTED": "已拒绝",
    "PROCESSING": "处理中",
    "COMPLETED": "已完成",
    "CANCELLED": "已取消",
}

def _build_order_list_message(order_status: OrderStatusValue | None) -> str:
    """根据订单状态生成给用户看的打开页面提示语。"""

    if order_status is None:
        return "已为你打开订单列表"
    status_label = _ORDER_STATUS_LABELS.get(order_status, order_status)
    return f"已为你打开{status_label}订单列表"


def _build_after_sale_list_message(
        after_sale_status: AfterSaleStatusValue | None,
) -> str:
    """根据售后状态生成给用户看的打开页面提示语。"""

    if after_sale_status is None:
        return "已为你打开售后列表"
    status_label = _AFTER_SALE_STATUS_LABELS.get(after_sale_status, after_sale_status)
    return f"已为你打开{status_label}售后列表"


def _build_order_list_action_response(
        order_status: OrderStatusValue | None,
) -> AssistantResponse:
    """构建“打开订单列表”动作响应。"""

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


def _build_after_sale_list_action_response(
        after_sale_status: AfterSaleStatusValue | None,
) -> AssistantResponse:
    """构建“打开售后列表”动作响应。"""

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
    args_schema=OpenUserOrderListRequest,
    description=(
            "打开用户订单列表页面。"
            "调用时机：用户明确要求打开、进入或查看订单列表时。"
            "如果用户提到了待支付、待发货、待收货、已完成、已取消等状态，"
            "请传入对应的 orderStatus；否则不要传。"
    ),
)
async def open_user_order_list(orderStatus: OrderStatusValue | None = None) -> str:
    """将“打开订单列表”动作加入当前请求的最终 SSE 响应队列。"""

    response = _build_order_list_action_response(orderStatus)
    enqueue_final_sse_response(response)
    return _build_order_list_message(orderStatus)


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
    """将“打开售后列表”动作加入当前请求的最终 SSE 响应队列。"""

    response = _build_after_sale_list_action_response(afterSaleStatus)
    enqueue_final_sse_response(response)
    return _build_after_sale_list_message(afterSaleStatus)


__all__ = [
    "OpenUserAfterSaleListRequest",
    "OpenUserOrderListRequest",
    "open_user_after_sale_list",
    "open_user_order_list",
]
