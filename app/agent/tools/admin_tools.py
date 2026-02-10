from typing import Annotated, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.schemas.http_response import HttpResponse
from app.utils.http_client import HttpClient

# --- Pydantic 模式定义 (用于复杂查询) ---

class MallProductListQueryRequest(BaseModel):
    """商城商品列表查询请求参数"""
    page_num: Optional[int] = Field(default=1, description="页码（从 1 开始）")
    page_size: Optional[int] = Field(default=10, description="每页数量")
    id: Optional[int] = Field(default=None, description="商品ID")
    name: Optional[str] = Field(default=None, description="商品名称关键词")
    category_id: Optional[int] = Field(default=None, description="商品分类ID")
    status: Optional[int] = Field(default=None, description="状态（1-上架，0-下架）")
    min_price: Optional[float] = Field(default=None, description="最低价格")
    max_price: Optional[float] = Field(default=None, description="最高价格")

class MallOrderListRequest(BaseModel):
    """商城订单列表查询请求参数"""
    page_num: Optional[int] = Field(default=1, description="页码（从 1 开始）")
    page_size: Optional[int] = Field(default=10, description="每页数量")
    order_no: Optional[str] = Field(default=None, description="订单编号")
    pay_type: Optional[str] = Field(default=None, description="支付方式编码")
    order_status: Optional[str] = Field(default=None, description="订单状态编码")
    delivery_type: Optional[str] = Field(default=None, description="配送方式编码")
    receiver_name: Optional[str] = Field(default=None, description="收货人姓名")
    receiver_phone: Optional[str] = Field(default=None, description="收货人电话")

# --- 工具定义 ---

@tool
async def get_user_info() -> dict:
    """
    获取当前登录用户的基本信息。
    当用户询问“我是谁”、“我的账户信息”或需要用户ID进行后续操作时调用。
    """
    async with HttpClient() as client:
        response = await client.get(url="/agent/tools/current_user")
        return HttpResponse.parse_data(response)

@tool(args_schema=MallProductListQueryRequest)
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
        response = await client.get(url="/agent/tools/products/search", params=params)
        return HttpResponse.parse_data(response)

@tool
async def get_product_info(
    product_id: Annotated[str, "商品的唯一ID，例如 '1001'"]
) -> dict:
    """
    根据商品ID获取详细的商品信息，包括描述、库存和规格。
    """
    async with HttpClient() as client:
        response = await client.get(url=f"/products/{product_id}")
        return HttpResponse.parse_data(response)

@tool(args_schema=MallOrderListRequest)
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
        response = await client.get(url="/agent/tools/orders/list", params=params)
        return HttpResponse.parse_data(response)

@tool
async def get_orders_detail(
    order_id: Annotated[str, "订单的唯一编号"]
) -> dict:
    """
    获取单个订单的详细信息，包括物流状态和商品明细。
    """
    async with HttpClient() as client:
        response = await client.get(url=f"/agent/tools/orders/{order_id}")
        return HttpResponse.parse_data(response)

# --- 工具集导出 ---

ADMIN_TOOLS = [
    get_user_info,
    get_product_list,
    get_product_info,
    get_order_list,
    get_orders_detail,
]