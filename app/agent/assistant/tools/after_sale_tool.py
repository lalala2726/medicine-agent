from __future__ import annotations

from typing import Optional

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from app.agent.assistant.tools.schemas.after_sale import (
    AdminAfterSaleIdRequest,
    AdminAfterSaleListQueryRequest,
)
from app.core.agent.agent_runtime import agent_invoke
from app.core.agent.agent_tool_events import tool_call_status
from app.core.langsmith import traceable
from app.core.llms import create_chat_model
from app.schemas.http_response import HttpResponse
from app.utils.http_client import HttpClient
from app.utils.prompt_utils import load_prompt


@tool(
    args_schema=AdminAfterSaleListQueryRequest,
    description=(
            "分页查询管理端售后申请列表，支持按售后类型、售后状态、订单号、用户 ID、申请原因筛选。"
            "参数传递规则：使用结构化字段，不要把多个筛选条件拼成单字符串。"
    ),
)
@tool_call_status(
    tool_name="查询售后列表",
    start_message="正在查询售后列表",
    error_message="查询售后列表失败",
    timely_message="售后列表正在持续处理中",
)
async def get_admin_after_sale_list(
        page_num: int = 1,
        page_size: int = 10,
        after_sale_type: Optional[str] = None,
        after_sale_status: Optional[str] = None,
        order_no: Optional[str] = None,
        user_id: Optional[int] = None,
        apply_reason: Optional[str] = None,
) -> dict:
    """分页查询管理端售后申请列表。"""

    async with HttpClient() as client:
        params = {
            "pageNum": page_num,
            "pageSize": page_size,
            "afterSaleType": after_sale_type,
            "afterSaleStatus": after_sale_status,
            "orderNo": order_no,
            "userId": user_id,
            "applyReason": apply_reason,
        }
        response = await client.get(
            url="/agent/admin/after-sale/list",
            params=params,
        )
        return HttpResponse.parse_data(response)


@tool(
    args_schema=AdminAfterSaleIdRequest,
    description=(
            "根据售后申请 ID 查询售后详情。"
            "调用时机：需要查看某个售后申请的处理进度、原因、处理结果或关联订单信息时。"
    ),
)
@tool_call_status(
    tool_name="查询售后详情",
    start_message="正在查询售后详情",
    error_message="查询售后详情失败",
    timely_message="售后详情正在持续处理中",
)
async def get_admin_after_sale_detail(after_sale_id: int) -> dict:
    """根据售后申请 ID 查询售后详情。"""

    async with HttpClient() as client:
        response = await client.get(
            url=f"/agent/admin/after-sale/detail/{after_sale_id}",
        )
        return HttpResponse.parse_data(response)


_AFTER_SALE_SYSTEM_PROMPT = load_prompt("assistant/after_sale_system_prompt.md")


@tool(
    description=(
            "处理管理端售后相关任务：售后列表、售后详情。"
            "输入为自然语言任务描述，内部会自动调用售后域工具并返回结果。"
    )
)
@tool_call_status(
    tool_name="正在调用售后子代理",
    start_message="正在执行售后查询",
    error_message="调用售后子代理失败",
    timely_message="售后子代理正在持续处理中",
)
@traceable(name="Supervisor After Sale Tool Agent", run_type="chain")
def after_sale_tool_agent(task_description: str) -> str:
    llm = create_chat_model(
        model="qwen-flash",
        temperature=1.0,
    )
    agent = create_agent(
        model=llm,
        system_prompt=SystemMessage(content=_AFTER_SALE_SYSTEM_PROMPT),
        tools=[
            get_admin_after_sale_list,
            get_admin_after_sale_detail,
        ],
    )
    input_messages = str(task_description or "").strip()
    result = agent_invoke(
        agent,
        input_messages,
    )
    content = str(result.content or "").strip()
    return content or "暂无数据"
