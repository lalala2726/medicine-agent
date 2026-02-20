from __future__ import annotations

import json
from typing import Any

from app.agent.admin.agent_utils import (
    build_mode_aware_instruction_payload,
    run_standard_tool_worker,
)
from app.agent.admin.state import AgentState
from app.agent.admin.tools.admin_tools import get_order_list, get_orders_detail
from app.core.assistant_status import status_node
from app.core.langsmith import traceable
from app.schemas.prompt import base_prompt

# 订单节点系统提示词：约束该节点只处理订单域任务，并按执行模式使用历史或主管指令。
_ORDER_SYSTEM_PROMPT = (
        """
你是药品商城后台的订单节点（order_agent）。
只处理订单相关任务，不要处理商品/闲聊。

输入约定：
1. 你收到的是 instruction JSON，包含 user_input/context/execution_mode/task_difficulty。
2. execution_mode=supervisor_loop 时，必须优先执行 directive。
3. execution_mode=fast_lane 时，可结合 chat_history 理解上下文。

核心规则：
1. 只基于真实工具结果回答，不得编造。
2. 优先复用 context（如 extracted_order_ids），不要重复追问已提供信息。
3. 若需要查询订单详情，必须保证 get_orders_detail 的 order_id 为 List[str] JSON 数组，
   不能传逗号拼接字符串。
4. 有批量 ID 时优先批量查询，不要逐条重复调用。
5. 输出要直接、简洁，不要重复相同段落，不要输出“我将调用某工具”这类过程性描述。
6. 无结果时明确说“未查到相关订单信息”并给出下一步建议（如果有）。
7. 若能从订单明细提取商品ID，请在结果中明确列出，便于后续商品节点继续处理。

示例 A（fast_lane）：
instruction:
{
  "user_input": "查一下订单 O20260101",
  "execution_mode": "fast_lane",
  "context": {}
}
期望行为：
调用 get_orders_detail，参数示例 {"order_id":["O20260101"]}，返回订单关键信息。

示例 B（supervisor_loop，延续查询）：
instruction:
{
  "execution_mode": "supervisor_loop",
  "directive": "查询这10个订单的收货人信息并提取商品ID",
  "context": {"extracted_order_ids":["O1","O2","O3"]}
}
期望行为：
直接基于 context.extracted_order_ids 调用 get_orders_detail（List[str]），
返回收货人信息并在文本中列出可提取的商品ID。

示例 C（模糊输入但上下文明确）：
instruction:
{
  "user_input": "继续查",
  "execution_mode": "fast_lane",
  "context": {"extracted_order_ids":["O1","O2"]}
}
期望行为：
延续查询 O1/O2 的订单详情，不重复向用户要订单号。

输入 instruction 是 JSON。
"""
        + base_prompt
)


def _build_instruction(state: AgentState) -> str:
    """
    生成订单节点 instruction JSON 字符串。

    该函数保留给现有测试与调试调用，同时内部复用公共 helper，
    保证与 `product_node` 的构建逻辑一致。

    Args:
        state: 当前图状态，需包含 `routing.mode` 以及可选的 `messages/context/directive`。

    Returns:
        str: JSON 字符串形式的 instruction，字段由
            `build_mode_aware_instruction_payload` 统一生成。
    """

    payload = build_mode_aware_instruction_payload(state)
    return json.dumps(payload, ensure_ascii=False, default=str)


@status_node(
    node="order",
    start_message="正在处理订单问题",
    display_when="always",
)
@traceable(name="Supervisor Order Agent Node", run_type="chain")
def order_agent(state: AgentState) -> dict[str, Any]:
    """
    执行订单 worker 节点并返回标准状态更新。

    Args:
        state: 当前 LangGraph 状态，包含用户输入、上下文、路由模式与历史消息。

    Returns:
        dict[str, Any]: 节点增量更新结果，至少包含：
            - `results.order`：订单节点输出内容与是否结束标志；
            - `context`：合并后的共享上下文（含提取 ID 与最新节点输出）；
            - `messages`：一条 AI 回复消息；
            - `execution_traces`：执行链路追踪信息；
            - `errors`（可选）：失败时的错误说明。
    """

    return run_standard_tool_worker(
        state=state,
        node_name="order_agent",
        result_key="order",
        system_prompt=_ORDER_SYSTEM_PROMPT,
        tools=[get_order_list, get_orders_detail],
        fallback_content="订单服务暂时不可用，请稍后重试。",
        fallback_error="order_agent 执行失败",
        enable_stream=True,
        failure_policy={},
    )
