from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from app.agent.admin.tools import get_product_detail, get_product_list
from app.core.langsmith import traceable
from app.core.llm import create_chat_model
from app.schemas.prompt import base_prompt
from app.utils.streaming_utils import invoke_with_trace

_PRODUCT_SYSTEM_PROMPT = (
        """
        你是商品域子工具（product_tool_agent），只负责商品相关问题。

        你只能处理：
        1. 商品列表查询（筛选、分页、上/下架、价格区间）。
        2. 商品详情查询（指定商品ID）。

        强约束：
        1. 你必须优先通过工具获取真实数据，严禁编造商品信息。
        2. 工具参数必须结构化，严格遵守工具 schema。
        3. 用户信息不足时，先给最小补充建议或说明缺少的关键参数。
        4. 输出必须简洁，优先给结论，再给必要字段。
        """
        + base_prompt
)


@tool(
    description=(
            "处理商品相关任务：商品列表、商品详情。"
            "输入为自然语言任务描述，内部会自动调用商品工具并返回结果。"
    )
)
@traceable(name="Supervisor Product Tool Agent", run_type="chain")
def product_tool_agent(task_description: str) -> str:
    llm = create_chat_model(
        model="qwen-flash",
        temperature=0.2,
    )
    messages = [
        SystemMessage(content=_PRODUCT_SYSTEM_PROMPT),
        HumanMessage(content=str(task_description or "").strip()),
    ]
    trace = invoke_with_trace(
        llm,
        messages,
        tools=[get_product_list, get_product_detail],
    )
    text = str(trace.get("text") or "").strip()
    if not text:
        return "未获取到商品数据，请补充筛选条件后重试。"
    return text


# 兼容旧引用：保留旧名称别名，避免其他模块导入中断。
product_agent = product_tool_agent
