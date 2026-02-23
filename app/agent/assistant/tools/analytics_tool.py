from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.utils.prompt_utils import load_prompt
from app.core.agent.agent_tool_events import tool_call_status
from app.core.agent.agent_runtime import run_agent_invoke_with_trace
from app.core.langsmith import traceable
from app.core.llm import create_agent_instance
from app.schemas.http_response import HttpResponse
from app.utils.http_client import HttpClient


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


@tool(
    description=(
        "获取运营总览数据。"
        "包括总订单数、总销售额、总用户数、退款统计等关键指标。"
    )
)
@tool_call_status(
    tool_name="运营总览",
    start_message="正在查询运营总览",
    error_message="获取运营总览失败",
    timely_message="运营总览正在持续处理中",
)
async def get_analytics_overview() -> dict:
    """获取运营总览。"""

    async with HttpClient() as client:
        response = await client.get(url="/agent/analytics/overview")
        return HttpResponse.parse_data(response)


@tool(
    args_schema=AnalyticsOrderTrendRequest,
    description=(
        "获取订单趋势数据。"
        "根据 period 统计订单数量和金额变化趋势，"
        "period 支持 DAY/WEEK/MONTH。"
    ),
)
@tool_call_status(
    tool_name="订单趋势",
    start_message="正在查询订单趋势",
    error_message="获取订单趋势失败",
    timely_message="订单趋势正在持续处理中",
)
async def get_analytics_order_trend(period: Literal["DAY", "WEEK", "MONTH"] = "DAY") -> dict:
    """获取订单趋势。"""

    async with HttpClient() as client:
        response = await client.get(
            url="/agent/analytics/order/trend",
            params={"period": period},
        )
        return HttpResponse.parse_data(response)


@tool(
    description=(
        "获取订单状态分布。"
        "统计不同状态订单（待付款、待发货、已完成等）的数量和占比。"
    )
)
@tool_call_status(
    tool_name="订单状态分布",
    start_message="正在查询订单状态分布",
    error_message="获取订单状态分布失败",
    timely_message="订单状态分布正在持续处理中",
)
async def get_analytics_order_status_distribution() -> dict:
    """获取订单状态分布。"""

    async with HttpClient() as client:
        response = await client.get(url="/agent/analytics/order/status-distribution")
        return HttpResponse.parse_data(response)


@tool(
    description=(
        "获取支付方式分布。"
        "统计不同支付方式（支付宝、微信等）的使用次数和占比。"
    )
)
@tool_call_status(
    tool_name="支付方式分布",
    start_message="正在查询支付方式分布",
    error_message="获取支付方式分布失败",
    timely_message="支付方式分布正在持续处理中",
)
async def get_analytics_payment_distribution() -> dict:
    """获取支付方式分布。"""

    async with HttpClient() as client:
        response = await client.get(url="/agent/analytics/order/payment-distribution")
        return HttpResponse.parse_data(response)


@tool(
    args_schema=AnalyticsTopLimitRequest,
    description=(
        "获取热销商品排行榜。"
        "根据销量统计最受欢迎的商品，按销量降序返回，"
        "limit 默认10。"
    ),
)
@tool_call_status(
    tool_name="热销商品排行",
    start_message="正在查询热销商品排行",
    error_message="获取热销商品排行失败",
    timely_message="热销商品排行正在持续处理中",
)
async def get_analytics_hot_products(limit: int = 10) -> dict:
    """获取热销商品排行榜。"""

    async with HttpClient() as client:
        response = await client.get(
            url="/agent/analytics/product/hot",
            params={"limit": limit},
        )
        return HttpResponse.parse_data(response)


@tool(
    args_schema=AnalyticsTopLimitRequest,
    description=(
        "获取商品退货率统计。"
        "统计商品退货情况并按退货率降序返回，"
        "limit 默认10。"
    ),
)
@tool_call_status(
    tool_name="商品退货率",
    start_message="正在查询商品退货率",
    error_message="获取商品退货率失败",
    timely_message="商品退货率正在持续处理中",
)
async def get_analytics_product_return_rates(limit: int = 10) -> dict:
    """获取商品退货率统计。"""

    async with HttpClient() as client:
        response = await client.get(
            url="/agent/analytics/product/return-rate",
            params={"limit": limit},
        )
        return HttpResponse.parse_data(response)


_BASE_PROMPT = load_prompt("assistant_base_prompt")
_ANALYTICS_SYSTEM_PROMPT = load_prompt("assistant_analytics_system_prompt") + _BASE_PROMPT


@tool(
    description=(
        "处理运营分析相关任务：总览、趋势、分布、排行榜。"
        "输入为自然语言任务描述，内部会自动调用分析工具并返回结果。"
    )
)
@tool_call_status(
    tool_name="正在调用运营分析子代理",
    start_message="正在执行查询",
    error_message="调用运营分析子代理失败",
    timely_message="运营分析子代理正在持续处理中",
)
@traceable(name="Supervisor Analytics Tool Agent", run_type="chain")
def analytics_tool_agent(task_description: str) -> str:
    agent = create_agent_instance(
        model="qwen-flash",
        llm_kwargs={"temperature": 0.2},
        system_prompt=SystemMessage(content=_ANALYTICS_SYSTEM_PROMPT),
        tools=[
            get_analytics_overview,
            get_analytics_order_trend,
            get_analytics_order_status_distribution,
            get_analytics_payment_distribution,
            get_analytics_hot_products,
            get_analytics_product_return_rates,
        ],
    )
    trace = run_agent_invoke_with_trace(
        agent,
        [HumanMessage(content=str(task_description or "").strip())],
    )
    text = str(trace.get("text") or "").strip()
    if not text:
        return "未获取到运营分析数据，请补充查询维度或条件后重试。"
    return text


analytics_agent = analytics_tool_agent
