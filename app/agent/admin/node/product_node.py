from __future__ import annotations

import json
from typing import Any

from app.agent.admin.agent_utils import (
    build_mode_aware_instruction_payload,
    run_standard_tool_worker,
)
from app.agent.admin.state import AgentState
from app.agent.admin.tools.admin_tools import (
    get_drug_detail,
    get_product_detail,
    get_product_list,
)
from app.core.assistant_status import status_node
from app.core.langsmith import traceable
from app.schemas.prompt import base_prompt

# 商品节点系统提示词：约束该节点只处理商品域任务，并按执行模式使用历史或主管指令。
_PRODUCT_SYSTEM_PROMPT = (
        """
你是药品商城后台的商品节点（product_agent）。
只处理商品相关任务，不要处理订单/闲聊。

输入约定：
1. 你收到的是 instruction JSON，包含 user_input/context/execution_mode/task_difficulty。
2. execution_mode=supervisor_loop 时，必须优先执行 directive。
3. execution_mode=fast_lane 时，可结合 chat_history 理解上下文。

核心规则：
1. 只基于真实工具结果回答，不得编造。
2. 优先使用 context 中已提取 ID（如 extracted_product_ids），避免重复提问。
3. 调用 get_product_detail/get_drug_detail 时，product_id 必须为 List[str] JSON 数组。
4. 有多个商品ID时优先批量查询，不要逐个重复调用。
5. 输出要直接、简洁，不要重复段落，不要输出“我将调用某工具”。
6. 若无结果，明确返回“未查到相关商品信息”，并给出可执行下一步建议。
7. 若用户目标涉及上下架/库存判断，返回时优先包含状态字段和结论。

示例 A（fast_lane）：
instruction:
{
  "user_input": "查商品2001库存和上下架状态",
  "execution_mode": "fast_lane",
  "context": {}
}
期望行为：
调用商品详情相关工具，参数示例 {"product_id":["2001"]}，返回库存与状态。

示例 B（supervisor_loop，跨域第二步）：
instruction:
{
  "execution_mode": "supervisor_loop",
  "directive": "基于context.extracted_product_ids批量查询商品详情",
  "context": {"extracted_product_ids":["2001","2003"]}
}
期望行为：
批量调用详情工具，输出商品关键信息并给出汇总结论。

示例 C（药品详情）：
instruction:
{
  "user_input": "查这些商品的药品说明",
  "context": {"extracted_product_ids":["2001","2003"]},
  "execution_mode": "supervisor_loop",
  "directive": "查询药品说明书信息"
}
期望行为：
调用 get_drug_detail，参数 {"product_id":["2001","2003"]}，输出适应症/用法用量等要点。

输入 instruction 是 JSON。
"""
        + base_prompt
)


def _build_instruction(state: AgentState) -> str:
    """
    生成商品节点 instruction JSON 字符串。

    该函数保留给现有测试与调试调用，同时内部复用公共 helper，
    保证与 `order_node` 的构建逻辑一致。

    Args:
        state: 当前图状态，需包含 `routing.mode` 以及可选的 `messages/context/directive`。

    Returns:
        str: JSON 字符串形式的 instruction，字段由
            `build_mode_aware_instruction_payload` 统一生成。
    """

    payload = build_mode_aware_instruction_payload(state)
    return json.dumps(payload, ensure_ascii=False, default=str)


@status_node(
    node="product",
    start_message="正在处理商品问题",
    display_when="always",
)
@traceable(name="Supervisor Product Agent Node", run_type="chain")
def product_agent(state: AgentState) -> dict[str, Any]:
    """
    执行商品 worker 节点并返回标准状态更新。

    Args:
        state: 当前 LangGraph 状态，包含用户输入、上下文、路由模式与历史消息。

    Returns:
        dict[str, Any]: 节点增量更新结果，至少包含：
            - `results.product`：商品节点输出内容与是否结束标志；
            - `context`：合并后的共享上下文（含提取 ID 与最新节点输出）；
            - `messages`：一条 AI 回复消息；
            - `execution_traces`：执行链路追踪信息；
            - `errors`（可选）：失败时的错误说明。
    """

    return run_standard_tool_worker(
        state=state,
        node_name="product_agent",
        result_key="product",
        system_prompt=_PRODUCT_SYSTEM_PROMPT,
        tools=[get_product_list, get_product_detail, get_drug_detail],
        fallback_content="商品服务暂时不可用，请稍后重试。",
        fallback_error="product_agent 执行失败",
        enable_stream=True,
        failure_policy={},
    )
