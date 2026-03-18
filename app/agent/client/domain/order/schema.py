from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class OrderNoRequest(BaseModel):
    """按订单编号查询的请求参数。"""

    model_config = ConfigDict(extra="forbid")

    order_no: str = Field(..., min_length=1, description="订单编号")

    @field_validator("order_no")
    @classmethod
    def _normalize_order_no(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("order_no 不能为空")
        return normalized


__all__ = [
    "OrderNoRequest",
]
