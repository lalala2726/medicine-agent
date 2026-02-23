from __future__ import annotations

from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.utils.prompt_utils import load_prompt
from app.agent.assistant.tools.base_tools import _normalize_id_list, format_ids_to_string
from app.core.sse_tool_events import tool_call_status
from app.core.agent_trace import run_model_with_trace
from app.core.langsmith import traceable
from app.core.llm import create_chat_model
from app.schemas.http_response import HttpResponse
from app.utils.http_client import HttpClient


class MallProductListQueryRequest(BaseModel):
    """
    商城商品列表查询请求参数。

    传参说明：
    1. 至少传分页参数 `page_num/page_size`；
    2. 其余筛选参数按需传，不需要的字段不要传空字符串；
    3. 推荐示例：
       `{"page_num": 1, "page_size": 10, "name": "感冒", "status": 1}`。
    """

    page_num: Optional[int] = Field(default=1, description="页码，从 1 开始，默认为 1")
    page_size: Optional[int] = Field(default=10, description="每页数量，建议 10-50，默认为 10")
    id: Optional[int] = Field(default=None, description="商品ID，精确匹配单个商品")
    name: Optional[str] = Field(
        default=None,
        description="商品名称关键词，支持模糊搜索，例如 '感冒' 可匹配 '感冒灵颗粒'",
    )
    category_id: Optional[int] = Field(default=None, description="商品分类ID，用于筛选特定分类下的商品")
    status: Optional[int] = Field(default=None, description="商品状态筛选：1 表示上架商品，0 表示下架商品")
    min_price: Optional[float] = Field(default=None, description="最低价格，用于价格区间筛选，单位：元")
    max_price: Optional[float] = Field(default=None, description="最高价格，用于价格区间筛选，单位：元")


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


@tool(
    args_schema=MallProductListQueryRequest,
    description=(
        "查看商城商品列表，如果不传入任何参数，只传递分页信息的话，这边默认返回最新的前N条数据，"
        "支持按名称、价格区间、分类等条件筛选。"
        "调用时机：当用户关注于商城内的商品信息时。"
    ),
)
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
    max_price: Optional[float] = None,
) -> dict:
    """搜索商城商品列表。"""

    async with HttpClient() as client:
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
    """根据商品ID获取详细信息，支持批量查询。"""

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
    """根据药品商品ID获取详细药品信息。"""

    normalized_ids = _normalize_id_list(product_id, field_name="product_id")
    ids_str = format_ids_to_string(normalized_ids)
    async with HttpClient() as client:
        response = await client.get(url=f"/agent/product/drug/{ids_str}")
        return HttpResponse.parse_data(response)


_BASE_PROMPT = load_prompt("assistant_base_prompt")
_PRODUCT_SYSTEM_PROMPT = load_prompt("assistant_product_system_prompt") + _BASE_PROMPT


@tool(
    description=(
        "处理商品相关任务：商品列表、商品详情。"
        "输入为自然语言任务描述，内部会自动调用商品工具并返回结果。"
    )
)
@traceable(name="Supervisor Product Tool Agent", run_type="chain")
@tool_call_status(
    tool_name="正在调用商品子代理",
    start_message="正在执行查询",
    error_message="调用商品子代理失败",
    timely_message="商品子代理正在持续处理中",
)
def product_tool_agent(task_description: str) -> str:
    llm = create_chat_model(
        model="qwen-flash",
        temperature=0.2,
    )
    messages = [
        SystemMessage(content=_PRODUCT_SYSTEM_PROMPT),
        HumanMessage(content=str(task_description or "").strip()),
    ]
    trace = run_model_with_trace(
        llm,
        messages,
        tools=[get_product_list, get_product_detail, get_drug_detail],
    )
    text = str(trace.get("text") or "").strip()
    if not text:
        return "未获取到商品数据，请补充筛选条件后重试。"
    return text


product_agent = product_tool_agent
