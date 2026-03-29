from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.sse_response import OrderStatusValue


class OrderNoRequest(BaseModel):
    """
    功能描述：
        按订单编号查询的请求参数模型。

    参数说明：
        order_no (str): 订单编号。

    返回值：
        无（数据模型定义）。

    异常说明：
        ValueError: 当 `order_no` 为空时抛出。
    """

    model_config = ConfigDict(extra="forbid")

    order_no: str = Field(..., min_length=1, description="订单编号")

    @field_validator("order_no")
    @classmethod
    def normalize_order_no(cls, value: str) -> str:
        """
        功能描述：
            规范化订单编号。

        参数说明：
            value (str): 原始订单编号。

        返回值：
            str: 去除首尾空白后的订单编号。

        异常说明：
            ValueError: 当订单编号为空时抛出。
        """

        normalized = value.strip()
        if not normalized:
            raise ValueError("order_no 不能为空")
        return normalized


class OpenUserOrderListRequest(BaseModel):
    """
    功能描述：
        打开用户订单列表工具参数模型。

    参数说明：
        orderStatus (OrderStatusValue | None): 订单状态筛选值。

    返回值：
        无（数据模型定义）。

    异常说明：
        无。
    """

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


__all__ = [
    "OpenUserOrderListRequest",
    "OrderNoRequest",
]
