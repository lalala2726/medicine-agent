"""运营分析工具参数模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AnalyticsDaysRequest(BaseModel):
    """按最近天数查询的运营分析请求。"""

    days: int = Field(
        default=30,
        ge=1,
        le=730,
        description="最近天数，默认30，范围1-730",
    )


class AnalyticsRankRequest(AnalyticsDaysRequest):
    """排行榜查询请求。"""

    limit: int = Field(
        default=10,
        ge=1,
        le=20,
        description="返回数量限制，默认10，范围1-20",
    )


__all__ = [
    "AnalyticsDaysRequest",
    "AnalyticsRankRequest",
]
