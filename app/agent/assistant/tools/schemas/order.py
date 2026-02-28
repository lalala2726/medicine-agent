"""
订单工具参数模型。
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class MallOrderListRequest(BaseModel):
    """
    商城订单列表查询请求参数。

    传参说明：
    1. 至少传分页参数 `page_num/page_size`；
    2. 其余筛选字段按需传，不要构造无意义空值；
    3. 推荐示例：
       `{"page_num": 1, "page_size": 10, "receiver_name": "张三"}`。
    """

    page_num: Optional[int] = Field(default=1, description="页码，从 1 开始，默认为 1")
    page_size: Optional[int] = Field(default=10, description="每页数量，建议 10-50，默认为 10")
    order_no: Optional[str] = Field(
        default=None,
        description="订单编号，精确匹配，例如 'O2024010112345678'",
    )
    pay_type: Optional[str] = Field(
        default=None,
        description="支付方式编码，例如 'wechat' 表示微信支付，'alipay' 表示支付宝",
    )
    order_status: Optional[str] = Field(
        default=None,
        description="订单状态编码，例如 'pending' 待支付，'paid' 已支付，'shipped' 已发货，'completed' 已完成，'cancelled' 已取消",
    )
    delivery_type: Optional[str] = Field(
        default=None,
        description="配送方式编码，例如 'express' 快递配送，'pickup' 到店自提",
    )
    receiver_name: Optional[str] = Field(default=None, description="收货人姓名，支持模糊搜索")
    receiver_phone: Optional[str] = Field(default=None, description="收货人手机号码，精确匹配")


class OrderDetailRequest(BaseModel):
    """
    订单详情查询请求参数。

    传参示例：
    `{"order_id": ["O20260101", "O20260102"]}`
    """

    order_id: list[str] = Field(
        min_length=1,
        description=(
            "订单ID字符串数组（List[str]），支持批量查询。"
            "必须传 JSON 数组，不能传 'O1,O2' 这种字符串。"
        ),
        examples=[["O20260101"], ["O20260101", "O20260102"]],
    )


class OrderIdRequest(BaseModel):
    """按订单 ID 查询请求参数。"""

    order_id: int = Field(ge=1, description="订单 ID")


__all__ = [
    "MallOrderListRequest",
    "OrderDetailRequest",
    "OrderIdRequest",
]
