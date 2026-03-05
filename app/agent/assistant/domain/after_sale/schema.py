"""
售后工具参数模型。
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class AdminAfterSaleListQueryRequest(BaseModel):
    """管理端售后列表查询请求参数。"""

    page_num: int = Field(default=1, ge=1, description="页码，从 1 开始")
    page_size: int = Field(default=10, ge=1, le=200, description="每页数量，范围 1-200")
    after_sale_type: Optional[str] = Field(
        default=None,
        description="售后类型，例如 REFUND_ONLY/RETURN_REFUND/EXCHANGE",
    )
    after_sale_status: Optional[str] = Field(
        default=None,
        description="售后状态，例如 PENDING/APPROVED/REJECTED/PROCESSING/COMPLETED/CANCELLED",
    )
    order_no: Optional[str] = Field(default=None, description="订单编号，精确匹配")
    user_id: Optional[int] = Field(default=None, ge=1, description="用户 ID")
    apply_reason: Optional[str] = Field(default=None, description="申请原因，例如 DAMAGED")


class AdminAfterSaleIdRequest(BaseModel):
    """按售后申请 ID 查询请求参数。"""

    after_sale_id: int = Field(ge=1, description="售后申请 ID")


__all__ = [
    "AdminAfterSaleIdRequest",
    "AdminAfterSaleListQueryRequest",
]
