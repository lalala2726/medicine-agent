from __future__ import annotations

from typing import Optional

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from app.agent.assistant.tools.base_tools import _normalize_id_list, format_ids_to_string
from app.agent.assistant.tools.schemas.order import (
    MallOrderListRequest,
    OrderDetailRequest,
    OrderIdRequest,
)
from app.core.agent.agent_runtime import agent_invoke
from app.core.agent.agent_tool_events import tool_call_status
from app.core.langsmith import traceable
from app.core.llm import create_agent
from app.utils.http_client import HttpClient
from app.utils.prompt_utils import load_prompt


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
        response = await client.get(
            url="/agent/admin/order/list",
            params=params,
        )
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
        response = await client.get(
            url=f"/agent/admin/order/{ids_str}",
        )
        return HttpResponse.parse_data(response)


@tool(
    args_schema=OrderIdRequest,
    description=(
            "根据订单 ID 查询订单流程时间线。"
            "调用时机：用户需要查看订单状态推进过程、关键时间节点时。"
    ),
)
@tool_call_status(
    tool_name="获取订单流程",
    start_message="正在查询订单流程",
    error_message="获取订单流程失败",
    timely_message="订单流程正在持续处理中",
)
async def get_order_timeline(order_id: int) -> dict:
    """根据订单 ID 查询订单流程（时间线）。"""

    async with HttpClient() as client:
        response = await client.get(
            url=f"/agent/admin/order/timeline/{order_id}",
        )
        return HttpResponse.parse_data(response)


@tool(
    args_schema=OrderIdRequest,
    description=(
            "根据订单 ID 查询发货记录。"
            "调用时机：用户需要查看快递单号、发货时间、承运信息等时。"
    ),
)
@tool_call_status(
    tool_name="获取发货记录",
    start_message="正在查询发货记录",
    error_message="获取发货记录失败",
    timely_message="发货记录正在持续处理中",
)
async def get_order_shipping(order_id: int) -> dict:
    """根据订单 ID 查询发货记录。"""

    async with HttpClient() as client:
        response = await client.get(
            url=f"/agent/admin/order/shipping/{order_id}",
        )
        return HttpResponse.parse_data(response)


_ORDER_SYSTEM_PROMPT = load_prompt("assistant/order_system_prompt.md")


@tool(
    description=(
            "处理订单相关任务：订单列表、订单详情、订单流程、发货记录。"
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
        tools=[
            get_order_list,
            get_orders_detail,
            get_order_timeline,
            get_order_shipping,
        ],
    )
    input_messages = str(task_description or "").strip()
    result = agent_invoke(
        agent,
        input_messages,
    )
    content = str(result.content or "").strip()
    return content or "暂无数据"
