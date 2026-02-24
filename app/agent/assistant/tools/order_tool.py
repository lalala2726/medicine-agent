from __future__ import annotations

from typing import Optional

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.utils.prompt_utils import load_prompt
from app.agent.assistant.tools.base_tools import _normalize_id_list, format_ids_to_string
from app.core.agent.agent_tool_events import tool_call_status
from app.core.agent.agent_runtime import agent_invoke
from app.core.langsmith import traceable
from app.core.llm import create_agent
from app.schemas.http_response import HttpResponse
from app.utils.http_client import HttpClient


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


@tool(
    args_schema=MallOrderListRequest,
    description=(
        "获取订单列表，支持按订单号、状态、收货人信息等条件筛选。"
        "参数传递规则：使用结构化字段（如 order_no、receiver_name），"
        "不要把多个筛选条件拼成单字符串。"
        "注意：若用户需要收货地址、物流详情等明细，请调用 get_orders_detail。"
        "调用时机：用户需要浏览或搜索订单时。"
    ),
)
@tool_call_status(
    tool_name="获取订单列表",
    start_message="正在查询订单列表",
    error_message="获取订单列表失败",
    timely_message="订单列表正在持续处理中",
)
async def get_order_list(
    page_num: int = 1,
    page_size: int = 10,
    order_no: Optional[str] = None,
    pay_type: Optional[str] = None,
    order_status: Optional[str] = None,
    delivery_type: Optional[str] = None,
    receiver_name: Optional[str] = None,
    receiver_phone: Optional[str] = None,
) -> dict:
    """获取订单列表。"""

    async with HttpClient() as client:
        params = {
            "pageNum": page_num,
            "pageSize": page_size,
            "orderNo": order_no,
            "payType": pay_type,
            "orderStatus": order_status,
            "deliveryType": delivery_type,
            "receiverName": receiver_name,
            "receiverPhone": receiver_phone,
        }
        response = await client.get(url="/agent/order/list", params=params)
        return HttpResponse.parse_data(response)


@tool(
    args_schema=OrderDetailRequest,
    description=(
        "根据订单ID获取详细信息，包括收货地址、物流信息、商品明细等，支持批量查询。"
        "参数传递规则：order_id 必须是字符串数组 List[str]，例如 "
        "{\"order_id\": [\"O20260101\", \"O20260102\"]}；"
        "不要传 'O20260101,O20260102'。"
        "调用时机：用户询问订单明细，或订单列表信息不足时。"
    ),
)
@tool_call_status(
    tool_name="获取订单详情",
    start_message="正在查询订单详情",
    error_message="获取订单详情失败",
    timely_message="订单详情正在持续处理中",
)
async def get_orders_detail(order_id: list[str]) -> dict:
    """获取订单详情，支持批量查询。"""

    normalized_ids = _normalize_id_list(order_id, field_name="order_id")
    ids_str = format_ids_to_string(normalized_ids)
    async with HttpClient() as client:
        response = await client.get(url=f"/agent/order/{ids_str}")
        return HttpResponse.parse_data(response)


_BASE_PROMPT = load_prompt("assistant_base_prompt")
_ORDER_SYSTEM_PROMPT = load_prompt("assistant_order_system_prompt") + _BASE_PROMPT


@tool(
    description=(
        "处理订单相关任务：订单列表、订单详情。"
        "输入为自然语言任务描述，内部会自动调用订单工具并返回结果。"
    )
)
@tool_call_status(
    tool_name="正在调用订单子代理",
    start_message="正在执行查询",
    error_message="调用订单子代理失败",
    timely_message="订单子代理正在持续处理中",
)
@traceable(name="Supervisor Order Tool Agent", run_type="chain")
def order_tool_agent(task_description: str) -> str:
    agent = create_agent(
        model="qwen-flash",
        llm_kwargs={"temperature": 0.2},
        system_prompt=SystemMessage(content=_ORDER_SYSTEM_PROMPT),
        tools=[get_order_list, get_orders_detail],
    )
    result = agent_invoke(
        agent,
        task_description,
    )
    return result.content
