from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AfterSaleNoRequest(BaseModel):
    """按售后单号查询请求参数。"""

    model_config = ConfigDict(extra="forbid")

    after_sale_no: str = Field(..., min_length=1, description="售后单号")

    @field_validator("after_sale_no")
    @classmethod
    def _normalize_after_sale_no(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("after_sale_no 不能为空")
        return normalized


class AfterSaleEligibilityRequest(BaseModel):
    """售后资格校验请求参数。"""

    model_config = ConfigDict(extra="forbid")

    order_no: str = Field(..., min_length=1, description="订单编号")
    order_item_id: int | None = Field(
        default=None,
        ge=1,
        description="订单项 ID，不传表示校验整单",
    )

    @field_validator("order_no")
    @classmethod
    def _normalize_order_no(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("order_no 不能为空")
        return normalized


__all__ = [
    "AfterSaleEligibilityRequest",
    "AfterSaleNoRequest",
]
