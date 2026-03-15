from __future__ import annotations

from collections.abc import Mapping

from langchain_core.tools import tool

from app.agent.admin.domain.analytics.schema import (
    AnalyticsDaysRequest,
    AnalyticsRankRequest,
)
from app.core.agent.agent_tool_events import tool_call_status
from app.schemas.http_response import HttpResponse
from app.utils.http_client import HttpClient

_DAYS_HELP = (
    "days 为最近天数，默认 30，范围 1-730。"
    "常用值：7 表示近7天，15 表示近15天，30 表示近30天，84 表示近12周，365 表示近12月。"
)


async def _request_analytics_data(
        *,
        url: str,
        params: Mapping[str, object] | None = None,
) -> dict:
    """发送运营分析查询请求并返回统一解析后的数据。"""

    async with HttpClient() as client:
        response = await client.get(url=url, params=dict(params or {}))
        return HttpResponse.parse_data(response)


@tool(
    description=(
            "获取实时运营总览。"
            "包括累计成交、今日成交、待发货、待收货、待处理售后和处理中售后等实时指标。"
    )
)
@tool_call_status(
    tool_name="实时运营总览",
    start_message="正在查询实时运营总览",
    error_message="获取实时运营总览失败",
    timely_message="实时运营总览正在持续处理中",
)
async def get_analytics_realtime_overview() -> dict:
    """获取实时运营总览。"""

    return await _request_analytics_data(url="/agent/admin/analytics/realtime-overview")


@tool(
    args_schema=AnalyticsDaysRequest,
    description=(
            "获取经营结果汇总。"
            "返回成交金额、支付订单数、净成交额、退款金额、退款率和退货退款件数等指标。"
            f"{_DAYS_HELP}"
    ),
)
@tool_call_status(
    tool_name="经营结果汇总",
    start_message="正在查询经营结果汇总",
    error_message="获取经营结果汇总失败",
    timely_message="经营结果汇总正在持续处理中",
)
async def get_analytics_range_summary(days: int = 30) -> dict:
    """获取经营结果汇总。"""

    return await _request_analytics_data(
        url="/agent/admin/analytics/range-summary",
        params={"days": days},
    )


@tool(
    args_schema=AnalyticsDaysRequest,
    description=(
            "获取支付转化汇总。"
            "返回下单数、已支付数、待支付数、关闭数和支付转化率。"
            f"{_DAYS_HELP}"
    ),
)
@tool_call_status(
    tool_name="支付转化汇总",
    start_message="正在查询支付转化汇总",
    error_message="获取支付转化汇总失败",
    timely_message="支付转化汇总正在持续处理中",
)
async def get_analytics_conversion_summary(days: int = 30) -> dict:
    """获取支付转化汇总。"""

    return await _request_analytics_data(
        url="/agent/admin/analytics/conversion-summary",
        params={"days": days},
    )


@tool(
    args_schema=AnalyticsDaysRequest,
    description=(
            "获取履约时效汇总。"
            "返回平均发货时长、平均收货时长、超时未发货订单数和超时未收货订单数。"
            f"{_DAYS_HELP}"
    ),
)
@tool_call_status(
    tool_name="履约时效汇总",
    start_message="正在查询履约时效汇总",
    error_message="获取履约时效汇总失败",
    timely_message="履约时效汇总正在持续处理中",
)
async def get_analytics_fulfillment_summary(days: int = 30) -> dict:
    """获取履约时效汇总。"""

    return await _request_analytics_data(
        url="/agent/admin/analytics/fulfillment-summary",
        params={"days": days},
    )


@tool(
    args_schema=AnalyticsDaysRequest,
    description=(
            "获取售后处理时效汇总。"
            "返回平均审核时长、平均完结时长、超24小时未审核数和超72小时未完结数。"
            f"{_DAYS_HELP}"
    ),
)
@tool_call_status(
    tool_name="售后处理时效汇总",
    start_message="正在查询售后处理时效汇总",
    error_message="获取售后处理时效汇总失败",
    timely_message="售后处理时效汇总正在持续处理中",
)
async def get_analytics_after_sale_efficiency_summary(
        days: int = 30,
) -> dict:
    """获取售后处理时效汇总。"""

    return await _request_analytics_data(
        url="/agent/admin/analytics/after-sale-efficiency-summary",
        params={"days": days},
    )


@tool(
    args_schema=AnalyticsDaysRequest,
    description=(
            "获取售后状态分布。"
            "返回待审核、已通过、处理中、已完成等售后状态的数量分布。"
            f"{_DAYS_HELP}"
    ),
)
@tool_call_status(
    tool_name="售后状态分布",
    start_message="正在查询售后状态分布",
    error_message="获取售后状态分布失败",
    timely_message="售后状态分布正在持续处理中",
)
async def get_analytics_after_sale_status_distribution(
        days: int = 30,
) -> dict:
    """获取售后状态分布。"""

    return await _request_analytics_data(
        url="/agent/admin/analytics/after-sale-status-distribution",
        params={"days": days},
    )


@tool(
    args_schema=AnalyticsDaysRequest,
    description=(
            "获取售后原因分布。"
            "返回商品损坏、描述不符、不想要了等售后原因的数量分布。"
            f"{_DAYS_HELP}"
    ),
)
@tool_call_status(
    tool_name="售后原因分布",
    start_message="正在查询售后原因分布",
    error_message="获取售后原因分布失败",
    timely_message="售后原因分布正在持续处理中",
)
async def get_analytics_after_sale_reason_distribution(
        days: int = 30,
) -> dict:
    """获取售后原因分布。"""

    return await _request_analytics_data(
        url="/agent/admin/analytics/after-sale-reason-distribution",
        params={"days": days},
    )


@tool(
    args_schema=AnalyticsRankRequest,
    description=(
            "获取热销商品排行。"
            "按销量返回商品名称、商品图、销量和成交金额，limit 默认10，最大20。"
            f"{_DAYS_HELP}"
    ),
)
@tool_call_status(
    tool_name="热销商品排行",
    start_message="正在查询热销商品排行",
    error_message="获取热销商品排行失败",
    timely_message="热销商品排行正在持续处理中",
)
async def get_analytics_top_selling_products(
        days: int = 30,
        limit: int = 10,
) -> dict:
    """获取热销商品排行。"""

    return await _request_analytics_data(
        url="/agent/admin/analytics/top-selling-products",
        params={"days": days, "limit": limit},
    )


@tool(
    args_schema=AnalyticsRankRequest,
    description=(
            "获取退货退款风险商品排行。"
            "返回商品销量、退货退款件数、退货退款率和退款金额，limit 默认10，最大20。"
            f"{_DAYS_HELP}"
    ),
)
@tool_call_status(
    tool_name="退货退款风险商品排行",
    start_message="正在查询退货退款风险商品排行",
    error_message="获取退货退款风险商品排行失败",
    timely_message="退货退款风险商品排行正在持续处理中",
)
async def get_analytics_return_refund_risk_products(
        days: int = 30,
        limit: int = 10,
) -> dict:
    """获取退货退款风险商品排行。"""

    return await _request_analytics_data(
        url="/agent/admin/analytics/return-refund-risk-products",
        params={"days": days, "limit": limit},
    )


@tool(
    args_schema=AnalyticsDaysRequest,
    description=(
            "获取成交趋势。"
            "返回完整时间轴上的成交金额和支付订单数趋势点位，不包含售后字段。"
            "days 较大时后端会自动切到周或月粒度。"
            f"{_DAYS_HELP}"
    ),
)
@tool_call_status(
    tool_name="成交趋势",
    start_message="正在查询成交趋势",
    error_message="获取成交趋势失败",
    timely_message="成交趋势正在持续处理中",
)
async def get_analytics_sales_trend(days: int = 30) -> dict:
    """获取成交趋势。"""

    return await _request_analytics_data(
        url="/agent/admin/analytics/sales-trend",
        params={"days": days},
    )


@tool(
    args_schema=AnalyticsDaysRequest,
    description=(
            "获取售后趋势。"
            "返回完整时间轴上的退款金额和售后申请数趋势点位，不包含成交字段。"
            "days 较大时后端会自动切到周或月粒度。"
            f"{_DAYS_HELP}"
    ),
)
@tool_call_status(
    tool_name="售后趋势",
    start_message="正在查询售后趋势",
    error_message="获取售后趋势失败",
    timely_message="售后趋势正在持续处理中",
)
async def get_analytics_after_sale_trend(days: int = 30) -> dict:
    """获取售后趋势。"""

    return await _request_analytics_data(
        url="/agent/admin/analytics/after-sale-trend",
        params={"days": days},
    )
