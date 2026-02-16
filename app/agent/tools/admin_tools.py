from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.core.assistant_status import tool_call_status
from app.schemas.http_response import HttpResponse
from app.utils.http_client import HttpClient


def format_ids_to_string(ids: list[str]) -> str:
    """
    将 ID 列表转换为逗号分隔的字符串格式。

    Args:
        ids: ID 列表，如 ['1', '2', '3'] 或 ['26', '11', '15', '17']

    Returns:
        逗号分隔的字符串，如 '1,2,3' 或 '26,11,15,17'
    """
    return ",".join(str(id_) for id_ in ids)


class MallProductListQueryRequest(BaseModel):
    """商城商品列表查询请求参数"""
    page_num: Optional[int] = Field(
        default=1,
        description="页码，从 1 开始，默认为 1"
    )
    page_size: Optional[int] = Field(
        default=10,
        description="每页数量，建议 10-50，默认为 10"
    )
    id: Optional[int] = Field(
        default=None,
        description="商品ID，精确匹配单个商品"
    )
    name: Optional[str] = Field(
        default=None,
        description="商品名称关键词，支持模糊搜索，例如 '感冒' 可匹配 '感冒灵颗粒'"
    )
    category_id: Optional[int] = Field(
        default=None,
        description="商品分类ID，用于筛选特定分类下的商品"
    )
    status: Optional[int] = Field(
        default=None,
        description="商品状态筛选：1 表示上架商品，0 表示下架商品"
    )
    min_price: Optional[float] = Field(
        default=None,
        description="最低价格，用于价格区间筛选，单位：元"
    )
    max_price: Optional[float] = Field(
        default=None,
        description="最高价格，用于价格区间筛选，单位：元"
    )


class MallOrderListRequest(BaseModel):
    """商城订单列表查询请求参数"""
    page_num: Optional[int] = Field(
        default=1,
        description="页码，从 1 开始，默认为 1"
    )
    page_size: Optional[int] = Field(
        default=10,
        description="每页数量，建议 10-50，默认为 10"
    )
    order_no: Optional[str] = Field(
        default=None,
        description="订单编号，精确匹配，例如 'O2024010112345678'"
    )
    pay_type: Optional[str] = Field(
        default=None,
        description="支付方式编码，例如 'wechat' 表示微信支付，'alipay' 表示支付宝"
    )
    order_status: Optional[str] = Field(
        default=None,
        description="订单状态编码，例如 'pending' 待支付，'paid' 已支付，'shipped' 已发货，'completed' 已完成，'cancelled' 已取消"
    )
    delivery_type: Optional[str] = Field(
        default=None,
        description="配送方式编码，例如 'express' 快递配送，'pickup' 到店自提"
    )
    receiver_name: Optional[str] = Field(
        default=None,
        description="收货人姓名，支持模糊搜索"
    )
    receiver_phone: Optional[str] = Field(
        default=None,
        description="收货人手机号码，精确匹配"
    )


class ProductInfoRequest(BaseModel):
    """商品详情查询请求参数"""
    product_id: list[str] = Field(
        description="商品ID，支持单个或多个批量查询。"
    )


class DrugDetailRequest(BaseModel):
    """药品详情查询请求参数"""
    product_id: list[str] = Field(
        description="药品商品ID列表，支持批量查询。"
    )


class OrderDetailRequest(BaseModel):
    """订单详情查询请求参数"""
    order_id: list[str] = Field(
        description="订单ID，支持单个或多个批量查询。"
    )


@tool(description="获取当前登录用户的基本信息。"
                  "调用时机：用户询问「我是谁」「我的账户信息」，或需要用户ID进行后续操作时。")
@tool_call_status(
    tool_name="获取用户信息",
    start_message="正在获取用户信息",
    error_message="获取用户信息失败",
    timely_message="用户信息正在持续处理中",
)
async def get_user_info() -> dict:
    """
    获取当前登录用户的基本信息。
    当用户询问“我是谁”、“我的账户信息”或需要用户ID进行后续操作时调用。
    """
    async with HttpClient() as client:
        response = await client.get(url="/agent/tools/current_user")
        return HttpResponse.parse_data(response)


@tool(args_schema=MallProductListQueryRequest, description="查看商城商品列表，如果不传入任何参数，只传递分页信息的话，这边默认返回最新的前N条数据，"
                                                           "支持按名称、价格区间、分类等条件筛选。"
                                                           "调用时机：当用户关注于商城内的商品信息时。")
@tool_call_status(
    tool_name="获取商品列表",
    start_message="正在查询商品列表",
    error_message="获取商品列表失败",
    timely_message="商品列表正在持续处理中",
)
async def get_product_list(
        page_num: int = 1,
        page_size: int = 10,
        id: Optional[int] = None,
        name: Optional[str] = None,
        category_id: Optional[int] = None,
        status: Optional[int] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None
) -> dict:
    """
    搜索商城商品列表。可以根据名称、价格区间、分类等条件进行组合筛选。
    """
    async with HttpClient() as client:
        # 将 Python 的 snake_case 映射为 API 预期的参数名
        params = {
            "pageNum": page_num,
            "pageSize": page_size,
            "id": id,
            "name": name,
            "categoryId": category_id,
            "status": status,
            "minPrice": min_price,
            "maxPrice": max_price,
        }
        response = await client.get(url="/agent/tools/product/list", params=params)
        return HttpResponse.parse_data(response)


@tool(args_schema=ProductInfoRequest, description="根据商品ID获取详细信息，支持批量查询。"
                                                  "调用时机：用户明确询问某个或某些商品的详细信息时。")
@tool_call_status(
    tool_name="获取商品详情",
    start_message="正在查询商品详情",
    error_message="获取商品详情失败",
    timely_message="商品详情正在持续处理中",
)
async def get_product_detail(product_id: list[str]) -> dict:
    """
    根据商品ID获取详细信息，支持批量查询。
    后端路径格式：`/agent/products/{ids}`，例如 `/agent/tools/products/1001,1002`。
    """
    ids_str = format_ids_to_string(product_id)
    async with HttpClient() as client:
        response = await client.get(url=f"/agent/tools/product/{ids_str}")
        return HttpResponse.parse_data(response)


@tool(args_schema=DrugDetailRequest, description="根据商品ID获取药品详细信息，包括说明书、适应症、用法用量等，支持批量查询。"
                                                 "调用时机：用户询问药品的详细说明书、适应症、用法用量等信息时。")
@tool_call_status(
    tool_name="获取药品详情",
    start_message="正在查询药品详情",
    error_message="获取药品详情失败",
    timely_message="药品详情正在持续处理中",
)
async def get_drug_detail(product_id: list[str]) -> dict:
    """
    根据药品商品ID获取详细药品信息，包括药品说明书、适应症、用法用量等。
    支持批量查询多个药品。
    """
    ids_str = format_ids_to_string(product_id)
    async with HttpClient() as client:
        response = await client.get(url=f"/agent/tools/drug/{ids_str}")
        return HttpResponse.parse_data(response)


@tool(args_schema=MallOrderListRequest, description="获取订单列表，支持按订单号、状态、收货人信息等条件筛选。"
                                                    "注意：若用户需要更详细的订单信息（如收货地址、物流详情），请调用 get_orders_detail 工具。"
                                                    "调用时机：用户需要浏览或搜索订单时。")
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
        receiver_phone: Optional[str] = None
) -> dict:
    """
    获取订单列表。可以根据订单号、状态、收货人信息进行查询。
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
        response = await client.get(url="/agent/tools/order/list", params=params)
        return HttpResponse.parse_data(response)


@tool(args_schema=OrderDetailRequest, description="根据订单ID获取详细信息，包括收货地址、物流信息、商品明细等，支持批量查询。"
                                                  "调用时机：用户询问某个或某些订单的详细信息时，或订单列表信息无法满足用户需求时。")
@tool_call_status(
    tool_name="获取订单详情",
    start_message="正在查询订单详情",
    error_message="获取订单详情失败",
    timely_message="订单详情正在持续处理中",
)
async def get_orders_detail(order_id: list[str]) -> dict:
    """
    获取订单详情，支持批量查询。
    后端路径格式：`/agent/tools/orders/{ids}`，例如 `/agent/tools/orders/1,2,3`。
    """
    ids_str = format_ids_to_string(order_id)
    async with HttpClient() as client:
        response = await client.get(url=f"/agent/tools/order/{ids_str}")
        return HttpResponse.parse_data(response)


# --- 工具集导出 ---

ADMIN_TOOLS = [
    get_user_info,
    get_product_list,
    get_product_detail,
    get_order_list,
    get_orders_detail,
    get_drug_detail
]
