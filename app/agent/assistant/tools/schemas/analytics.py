"""
运营分析工具参数模型。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AnalyticsOrderTrendRequest(BaseModel):
    """订单趋势查询请求。"""

    period: Literal["DAY", "WEEK", "MONTH"] = Field(
        default="DAY",
        description="时间周期，支持 DAY(日)、WEEK(周)、MONTH(月)",
    )


class AnalyticsTopLimitRequest(BaseModel):
    """排行榜/统计 TopN 查询请求。"""

    limit: int = Field(
        default=10,
        ge=1,
        le=200,
        description="返回数量限制，默认10，范围1-200",
    )


__all__ = [
    "AnalyticsOrderTrendRequest",
    "AnalyticsTopLimitRequest",
]
