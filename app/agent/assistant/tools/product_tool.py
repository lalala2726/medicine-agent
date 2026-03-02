from __future__ import annotations

from typing import Optional

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from app.agent.assistant.tools.base_tools import _normalize_id_list, format_ids_to_string
from app.agent.assistant.tools.schemas.product import (
    DrugDetailRequest,
    MallProductListQueryRequest,
    ProductInfoRequest,
)
from app.core.agent.agent_runtime import agent_invoke
from app.core.agent.agent_tool_events import tool_call_status
from app.core.langsmith import traceable
from app.core.llms import create_chat_model
from app.schemas.http_response import HttpResponse
from app.utils.http_client import HttpClient
from app.utils.prompt_utils import load_prompt


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
        response = await client.get(url="/agent/admin/product/list", params=params)
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
        response = await client.get(url=f"/agent/admin/product/{ids_str}")
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
        response = await client.get(url=f"/agent/admin/product/drug/{ids_str}")
        return HttpResponse.parse_data(response)


_PRODUCT_SYSTEM_PROMPT = load_prompt("assistant/product_system_prompt.md")


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
        temperature=1.0,
    )
    agent = create_agent(
        model=llm,
        system_prompt=SystemMessage(content=_PRODUCT_SYSTEM_PROMPT),
        tools=[get_product_list, get_product_detail, get_drug_detail],
    )
    input_messages = str(task_description or "").strip()
    result = agent_invoke(
        agent,
        input_messages,
    )
    content = str(result.content or "").strip()
    return content or "暂无数据"
