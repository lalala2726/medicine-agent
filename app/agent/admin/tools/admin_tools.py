"""
管理后台工具集合（订单/商品/药品）。

约定：
1. 所有批量 ID 参数必须传 JSON 数组（List[str]），不能传逗号拼接字符串；
2. 所有参数命名以工具函数签名为准，Agent 调用时必须传同名字段；
3. 详情类工具在收到空 ID 或全空白 ID 时会直接报错并拒绝调用后端接口。
"""

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


def _normalize_id_list(ids: list[str], *, field_name: str) -> list[str]:
    """
    规范化并校验批量 ID 参数。

    Args:
        ids: 原始 ID 列表，要求元素可转字符串。
        field_name: 当前参数名，用于报错提示（如 `order_id`、`product_id`）。

    Returns:
        list[str]: 去首尾空白并移除空值后的 ID 列表。

    Raises:
        ValueError: 当入参不是非空字符串数组时抛出，防止请求路径拼成 `/agent/order/`。
    """

    normalized = [str(item).strip() for item in ids if str(item).strip()]
    if not normalized:
        raise ValueError(f"{field_name} 必须为非空字符串数组（List[str]），例如 [\"A1\",\"A2\"]")
    return normalized


class MallProductListQueryRequest(BaseModel):
    """
    商城商品列表查询请求参数。

    传参说明：
    1. 至少传分页参数 `page_num/page_size`；
    2. 其余筛选参数按需传，不需要的字段不要传空字符串；
    3. 推荐示例：
       `{"page_num": 1, "page_size": 10, "name": "感冒", "status": 1}`。
    """
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
    """
    商城订单列表查询请求参数。

    传参说明：
    1. 至少传分页参数 `page_num/page_size`；
    2. 其余筛选字段按需传，不要构造无意义空值；
    3. 推荐示例：
       `{"page_num": 1, "page_size": 10, "receiver_name": "张三"}`。
    """
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
    """
    商品详情查询请求参数。

    传参示例：
    `{"product_id": ["2001", "2003"]}`
    """

    product_id: list[str] = Field(
        min_length=1,
        description=(
            "商品ID字符串数组（List[str]），支持批量查询。"
            "必须传 JSON 数组，不能传 '2001,2003' 这种逗号字符串。"
        ),
        examples=[["2001"], ["2001", "2003"]],
    )


class DrugDetailRequest(BaseModel):
    """
    药品详情查询请求参数。

    传参示例：
    `{"product_id": ["2001", "2003"]}`
    """

    product_id: list[str] = Field(
        min_length=1,
        description=(
            "药品商品ID字符串数组（List[str]），支持批量查询。"
            "必须传 JSON 数组，不能传逗号拼接字符串。"
        ),
        examples=[["2001"], ["2001", "2003"]],
    )


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

    Args:
        无。

    Returns:
        dict: 用户信息字典（由后端 `/agent/info` 返回并解析）。
    """
    async with HttpClient() as client:
        response = await client.get(url="/agent/info")
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

    Args:
        page_num: 页码（从 1 开始）。
        page_size: 每页数量。
        id: 商品 ID 精确筛选。
        name: 商品名称关键词。
        category_id: 分类 ID。
        status: 上下架状态（1 上架，0 下架）。
        min_price: 最低价格。
        max_price: 最高价格。

    Returns:
        dict: 商品列表查询结果字典。
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
        response = await client.get(url="/agent/product/list", params=params)
        return HttpResponse.parse_data(response)


@tool(
    args_schema=ProductInfoRequest,
    description=(
        "根据商品ID获取详细信息，支持批量查询。"
        "参数传递规则：product_id 必须是字符串数组 List[str]，例如 "
        "{\"product_id\": [\"2001\", \"2003\"]}；"
        "不要传逗号拼接字符串。"
        "调用时机：用户明确询问某个或某些商品的详细信息时。"
    ),
)
@tool_call_status(
    tool_name="获取商品详情",
    start_message="正在查询商品详情",
    error_message="获取商品详情失败",
    timely_message="商品详情正在持续处理中",
)
async def get_product_detail(product_id: list[str]) -> dict:
    """
    根据商品ID获取详细信息，支持批量查询。
    后端路径格式：`/agent/product/{ids}`，例如 `/agent/product/1001,1002`。

    Args:
        product_id: 商品 ID 字符串数组（List[str]），例如 `["2001", "2003"]`。
            不能为空，不能传逗号拼接字符串。

    Returns:
        dict: 商品详情结果字典。
    """
    normalized_ids = _normalize_id_list(product_id, field_name="product_id")
    ids_str = format_ids_to_string(normalized_ids)
    async with HttpClient() as client:
        response = await client.get(url=f"/agent/product/{ids_str}")
        return HttpResponse.parse_data(response)


@tool(
    args_schema=DrugDetailRequest,
    description=(
        "根据商品ID获取药品详细信息，包括说明书、适应症、用法用量等，支持批量查询。"
        "参数传递规则：product_id 必须是字符串数组 List[str]，例如 "
        "{\"product_id\": [\"2001\", \"2003\"]}；"
        "不要传逗号拼接字符串。"
        "调用时机：用户询问药品说明书、适应症、用法用量等信息时。"
    ),
)
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

    Args:
        product_id: 药品商品 ID 字符串数组（List[str]），例如 `["2001", "2003"]`。
            不能为空，不能传逗号拼接字符串。

    Returns:
        dict: 药品详情结果字典。
    """
    normalized_ids = _normalize_id_list(product_id, field_name="product_id")
    ids_str = format_ids_to_string(normalized_ids)
    async with HttpClient() as client:
        response = await client.get(url=f"/agent/drug/{ids_str}")
        return HttpResponse.parse_data(response)


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
        receiver_phone: Optional[str] = None
) -> dict:
    """
    获取订单列表。可以根据订单号、状态、收货人信息进行查询。

    Args:
        page_num: 页码（从 1 开始）。
        page_size: 每页数量。
        order_no: 订单编号。
        pay_type: 支付方式编码。
        order_status: 订单状态编码。
        delivery_type: 配送方式编码。
        receiver_name: 收货人姓名。
        receiver_phone: 收货人手机号。

    Returns:
        dict: 订单列表查询结果字典。
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
    """
    获取订单详情，支持批量查询。
    后端路径格式：`/agent/order/{order_id}`，例如 `/agent/order/O20260101,O20260102`。

    Args:
        order_id: 订单ID 字符串数组（List[str]），例如 `["O20260101", "O20260102"]`。
            不能为空，不能传逗号拼接字符串。

    Returns:
        dict: 订单详情结果字典。
    """
    normalized_ids = _normalize_id_list(order_id, field_name="order_id")
    ids_str = format_ids_to_string(normalized_ids)
    async with HttpClient() as client:
        response = await client.get(url=f"/agent/order/{ids_str}")
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
