from __future__ import annotations

from typing import Optional

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.core.agent.agent_runtime import agent_invoke
from app.core.agent.agent_tool_events import tool_call_status
from app.core.langsmith import traceable
from app.core.llm import create_agent
from app.utils.http_client import HttpClient
from app.utils.prompt_utils import load_prompt


class AdminAfterSaleListQueryRequest(BaseModel):
    """管理端售后列表查询请求参数。"""

    page_num: int = Field(default=1, ge=1, description="页码，从 1 开始")
    page_size: int = Field(default=10, ge=1, le=200, description="每页数量，范围 1-200")
    after_sale_type: Optional[str] = Field(
        default=None,
        description="售后类型，例如 REFUND_ONLY/RETURN_REFUND/EXCHANGE",
    )
    after_sale_status: Optional[str] = Field(
        default=None,
        description="售后状态，例如 PENDING/APPROVED/REJECTED/PROCESSING/COMPLETED/CANCELLED",
    )
    order_no: Optional[str] = Field(default=None, description="订单编号，精确匹配")
    user_id: Optional[int] = Field(default=None, ge=1, description="用户 ID")
    apply_reason: Optional[str] = Field(default=None, description="申请原因，例如 DAMAGED")


class AdminAfterSaleIdRequest(BaseModel):
    """按售后申请 ID 查询请求参数。"""

    after_sale_id: int = Field(ge=1, description="售后申请 ID")


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
) -> str:
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
        return await client.get(
            url="/agent/admin/after-sale/list",
            params=params,
            response_format="yaml",
            include_envelope=True,
        )


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
async def get_admin_after_sale_detail(after_sale_id: int) -> str:
    """根据售后申请 ID 查询售后详情。"""

    async with HttpClient() as client:
        return await client.get(
            url=f"/agent/admin/after-sale/detail/{after_sale_id}",
            response_format="yaml",
            include_envelope=True,
        )


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
    agent = create_agent(
        model="qwen-flash",
        llm_kwargs={"temperature": 0.2},
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
