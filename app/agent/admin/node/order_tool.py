from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from app.agent.admin.tools import get_order_list, get_orders_detail
from app.core.langsmith import traceable
from app.core.llm import create_chat_model
from app.schemas.prompt import base_prompt
from app.utils.streaming_utils import invoke

# 订单节点系统提示词：约束该节点只处理订单域任务，并按执行模式使用历史或主管指令。
_ORDER_SYSTEM_PROMPT = (
        """
        你是订单域子工具（order_tool_agent），只负责订单相关问题。

        你只能处理：
        1. 订单列表查询（筛选、分页、状态、收货人条件）。
        2. 订单详情查询（指定订单ID，可批量）。

        强约束：
        1. 必须优先通过工具获取真实数据，严禁编造订单状态、金额、地址、物流信息。
        2. 当用户明确要“订单详情/收货地址/物流/商品明细”时，优先使用 get_orders_detail。
        3. 参数必须结构化并符合工具 schema，不要拼接自由文本参数。
        4. 输出简洁，先给结论，再给关键字段。
        """
        + base_prompt
)


@tool(
    description=(
        "处理订单相关任务：订单列表、订单详情。"
        "输入为自然语言任务描述，内部会自动调用订单工具并返回结果。"
    )
)
@traceable(name="Supervisor Order Tool Agent", run_type="chain")
def order_tool_agent(task_description: str) -> str:
    llm = create_chat_model(
        model="qwen-flash",
        temperature=0.2,
    )
    messages = [
        SystemMessage(content=_ORDER_SYSTEM_PROMPT),
        HumanMessage(content=str(task_description or "").strip()),
    ]
    content = invoke(
        llm,
        messages,
        tools=[get_order_list, get_orders_detail],
    )
    text = str(content or "").strip()
    if not text:
        return "未获取到订单数据，请补充订单号或筛选条件后重试。"
    return text


# 兼容旧引用：保留旧名称别名，避免其他模块导入中断。
order_agent = order_tool_agent
