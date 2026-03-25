"""
订单领域工具。
"""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.agent.admin.tools.base import format_ids_to_string, normalize_id_list
from app.core.agent.agent_tool_events import tool_call_status
from app.core.agent.tool_cache import ADMIN_TOOL_CACHE_PROFILE, tool_cacheable
from app.schemas.http_response import HttpResponse
from app.utils.http_client import HttpClient


class OrderListRequest(BaseModel):
    """
    功能描述：
        订单列表查询入参模型。

    参数说明：
        page_num (int | None): 页码。
        page_size (int | None): 每页数量。
        order_no (str | None): 订单号。
        pay_type (str | None): 支付方式。
        order_status (str | None): 订单状态。
        delivery_type (str | None): 配送方式。
        receiver_name (str | None): 收货人姓名。
        receiver_phone (str | None): 收货人手机号。

    返回值：
        无（数据模型定义）。

    异常说明：
        无。
    """

    page_num: Optional[int] = Field(default=1, description="页码，从 1 开始，默认为 1")
    page_size: Optional[int] = Field(default=10, description="每页数量，默认为 10")
    order_no: Optional[str] = Field(default=None, description="订单编号，精确匹配")
    pay_type: Optional[str] = Field(default=None, description="支付方式编码，例如 wechat、alipay")
    order_status: Optional[str] = Field(default=None, description="订单状态编码，例如 pending、paid、shipped")
    delivery_type: Optional[str] = Field(default=None, description="配送方式编码，例如 express、pickup")
    receiver_name: Optional[str] = Field(default=None, description="收货人姓名，支持模糊搜索")
    receiver_phone: Optional[str] = Field(default=None, description="收货人手机号，精确匹配")


class OrderDetailRequest(BaseModel):
    """
    功能描述：
        订单详情查询入参模型。

    参数说明：
        order_id (list[str]): 订单 ID 字符串数组。

    返回值：
        无（数据模型定义）。

    异常说明：
        无。
    """

    order_id: list[str] = Field(
        min_length=1,
        description="订单 ID 字符串数组，必须传 JSON 数组",
        examples=[["O20260101"], ["O20260101", "O20260102"]],
    )


class OrderIdRequest(BaseModel):
    """
    功能描述：
        单个订单 ID 查询入参模型。

    参数说明：
        order_id (int): 订单 ID。

    返回值：
        无（数据模型定义）。

    异常说明：
        无。
    """

    order_id: int = Field(ge=1, description="订单 ID")


def _build_order_list_cache_input(arguments: dict[str, object]) -> dict[str, object]:
    """
    功能描述：
        构造订单列表缓存入参。

    参数说明：
        arguments (dict[str, object]): 绑定后的函数参数映射。

    返回值：
        dict[str, object]: 与真实 HTTP 请求一致的查询参数结构。

    异常说明：
        无。
    """

    return {
        "pageNum": arguments.get("page_num"),
        "pageSize": arguments.get("page_size"),
        "orderNo": arguments.get("order_no"),
        "payType": arguments.get("pay_type"),
        "orderStatus": arguments.get("order_status"),
        "deliveryType": arguments.get("delivery_type"),
        "receiverName": arguments.get("receiver_name"),
        "receiverPhone": arguments.get("receiver_phone"),
    }


def _build_order_detail_cache_input(arguments: dict[str, object]) -> dict[str, object]:
    """
    功能描述：
        构造订单详情缓存入参。

    参数说明：
        arguments (dict[str, object]): 绑定后的函数参数映射。

    返回值：
        dict[str, object]: 标准化后的订单 ID 数组。

    异常说明：
        ValueError: 当 `order_id` 非法时抛出。
    """

    raw_order_id = arguments.get("order_id")
    normalized_ids = normalize_id_list(raw_order_id, field_name="order_id")
    return {"order_id": normalized_ids}


@tool(
    args_schema=OrderListRequest,
    description=(
            "分页查询订单列表，支持按订单号、状态、收货人和支付方式等条件筛选。"
            "适用于查订单列表、筛选订单或定位订单范围。"
    ),
)
@tool_call_status(
    tool_name="获取订单列表",
    start_message="正在查询订单列表",
    error_message="获取订单列表失败",
    timely_message="订单列表正在持续处理中",
)
@tool_cacheable(
    ADMIN_TOOL_CACHE_PROFILE,
    tool_name="order_list",
    input_builder=_build_order_list_cache_input,
)
async def order_list(
        page_num: int = 1,
        page_size: int = 10,
        order_no: Optional[str] = None,
        pay_type: Optional[str] = None,
        order_status: Optional[str] = None,
        delivery_type: Optional[str] = None,
        receiver_name: Optional[str] = None,
        receiver_phone: Optional[str] = None,
) -> dict:
    """
    功能描述：
        查询订单列表。

    参数说明：
        page_num (int): 页码。
        page_size (int): 每页数量。
        order_no (Optional[str]): 订单号。
        pay_type (Optional[str]): 支付方式。
        order_status (Optional[str]): 订单状态。
        delivery_type (Optional[str]): 配送方式。
        receiver_name (Optional[str]): 收货人姓名。
        receiver_phone (Optional[str]): 收货人手机号。

    返回值：
        dict: 订单列表数据。

    异常说明：
        无。
    """

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
            "根据订单 ID 数组查询订单详情。"
            "适用于查看订单收货地址、物流信息、商品明细和订单详细状态。"
    ),
)
@tool_call_status(
    tool_name="获取订单详情",
    start_message="正在查询订单详情",
    error_message="获取订单详情失败",
    timely_message="订单详情正在持续处理中",
)
@tool_cacheable(
    ADMIN_TOOL_CACHE_PROFILE,
    tool_name="order_detail",
    input_builder=_build_order_detail_cache_input,
)
async def order_detail(order_id: list[str]) -> dict:
    """
    功能描述：
        根据订单 ID 数组查询订单详情。

    参数说明：
        order_id (list[str]): 订单 ID 字符串数组。

    返回值：
        dict: 订单详情数据。

    异常说明：
        ValueError: 当 `order_id` 非法时抛出。
    """

    normalized_ids = normalize_id_list(order_id, field_name="order_id")
    ids_str = format_ids_to_string(normalized_ids)
    async with HttpClient() as client:
        response = await client.get(url=f"/agent/admin/order/{ids_str}")
        return HttpResponse.parse_data(response)


@tool(
    args_schema=OrderIdRequest,
    description=(
            "根据订单 ID 查询订单流程时间线。"
            "适用于查看订单状态推进过程和关键时间节点。"
    ),
)
@tool_call_status(
    tool_name="获取订单流程",
    start_message="正在查询订单流程",
    error_message="获取订单流程失败",
    timely_message="订单流程正在持续处理中",
)
@tool_cacheable(
    ADMIN_TOOL_CACHE_PROFILE,
    tool_name="order_timeline",
)
async def order_timeline(order_id: int) -> dict:
    """
    功能描述：
        根据订单 ID 查询订单流程时间线。

    参数说明：
        order_id (int): 订单 ID。

    返回值：
        dict: 订单流程时间线数据。

    异常说明：
        无。
    """

    async with HttpClient() as client:
        response = await client.get(url=f"/agent/admin/order/timeline/{order_id}")
        return HttpResponse.parse_data(response)


@tool(
    args_schema=OrderIdRequest,
    description=(
            "根据订单 ID 查询发货记录。"
            "适用于查看快递单号、发货时间和承运信息。"
    ),
)
@tool_call_status(
    tool_name="获取发货记录",
    start_message="正在查询发货记录",
    error_message="获取发货记录失败",
    timely_message="发货记录正在持续处理中",
)
@tool_cacheable(
    ADMIN_TOOL_CACHE_PROFILE,
    tool_name="order_shipping",
)
async def order_shipping(order_id: int) -> dict:
    """
    功能描述：
        根据订单 ID 查询发货记录。

    参数说明：
        order_id (int): 订单 ID。

    返回值：
        dict: 发货记录数据。

    异常说明：
        无。
    """

    async with HttpClient() as client:
        response = await client.get(url=f"/agent/admin/order/shipping/{order_id}")
        return HttpResponse.parse_data(response)


__all__ = [
    "OrderDetailRequest",
    "OrderIdRequest",
    "OrderListRequest",
    "order_detail",
    "order_list",
    "order_shipping",
    "order_timeline",
]
