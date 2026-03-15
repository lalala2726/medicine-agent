from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from langchain_core.tools import tool

from app.core.agent.agent_event_bus import enqueue_final_sse_response
from app.schemas.sse_response import (
    Action,
    AssistantResponse,
    Content,
    MessageType,
    OrderStatusValue,
    UserOrderListPayload,
)

_ORDER_STATUS_LABELS: dict[str, str] = {
    "PENDING_PAYMENT": "待支付",
    "PENDING_SHIPMENT": "待发货",
    "PENDING_RECEIPT": "待收货",
    "COMPLETED": "已完成",
    "CANCELLED": "已取消",
}
_DEFAULT_ACTION_PRIORITY = 100


class OpenUserOrderListRequest(BaseModel):
    """打开用户订单列表工具参数。"""

    model_config = ConfigDict(extra="forbid")

    orderStatus: OrderStatusValue | None = Field(
        default=None,
        description=(
            "订单状态，可选值："
            "PENDING_PAYMENT（待支付）、"
            "PENDING_SHIPMENT（待发货）、"
            "PENDING_RECEIPT（待收货）、"
            "COMPLETED（已完成）、"
            "CANCELLED（已取消）。"
        ),
    )


def _build_order_list_message(order_status: OrderStatusValue | None) -> str:
    """构造给前端展示与工具回执使用的动作文案。"""

    if order_status is None:
        return "已为你打开订单列表"
    status_label = _ORDER_STATUS_LABELS.get(order_status, order_status)
    return f"已为你打开{status_label}订单列表"


def _build_order_list_action_response(
        order_status: OrderStatusValue | None,
) -> AssistantResponse:
    """构造“打开用户订单列表”的最终动作 SSE 响应。"""

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
    """将“打开订单列表”动作加入当前请求的最终 SSE 响应队列。"""

    response = _build_order_list_action_response(orderStatus)
    enqueue_final_sse_response(response)
    return _build_order_list_message(orderStatus)


__all__ = [
    "OpenUserOrderListRequest",
    "open_user_order_list",
]
