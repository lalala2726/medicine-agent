from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.admin.agent_utils import build_execution_trace_update, serialize_messages
from app.agent.admin.model_policy import (
    NORMAL_DIFFICULTY,
    apply_model_profile_to_routing,
    build_gateway_decision,
    resolve_model_profile,
)
from app.agent.admin.state import AgentState
from app.core.langsmith import traceable
from app.core.llm import create_chat_model

_GATEWAY_PROMPT = """
你是药品商城后台助手的前置意图网关（Intent Gateway）。

你只能输出 JSON，且必须包含两个字段：
1) route_target: order_agent|product_agent|chat_agent|supervisor_agent
2) task_difficulty: simple|normal|complex

通用要求：
1. 只输出一个 JSON 对象，不要输出任何解释、Markdown、代码块。
2. 若用户输入很短（如“查询啊”），允许结合最近对话上下文延续同一业务域。
3. 若上下文不足以判断业务域，优先路由 chat_agent，避免误调用业务节点。

路由规则（优先级从高到低）：
1. 跨域、多步骤、依赖上下文决策、需要节点协作 -> supervisor_agent。
2. 单域订单任务（查订单/订单详情/收货人/退款单） -> order_agent。
3. 单域商品任务（查商品/库存/上下架/药品详情） -> product_agent。
4. 闲聊、寒暄、无明确业务动作 -> chat_agent。

难度规则：
1. simple: 单步、参数明确、直接查询。
2. normal: 需要少量推理或条件筛选。
3. complex: 跨域、多阶段、依赖上下文/策略判断。

示例：
- 用户: "查一下订单123"
  输出: {"route_target":"order_agent","task_difficulty":"simple"}
- 用户: "查商品2001库存"
  输出: {"route_target":"product_agent","task_difficulty":"simple"}
- 用户: "在吗"
  输出: {"route_target":"chat_agent","task_difficulty":"simple"}
- 用户: "把上个月退款超过2次的订单找出来，把对应商品下架"
  输出: {"route_target":"supervisor_agent","task_difficulty":"complex"}
- 上下文: 最近在查订单收货人，用户当前输入: "查询啊"
  输出: {"route_target":"order_agent","task_difficulty":"normal"}
"""


def _resolve_mode(route_target: str) -> str:
    """
    根据 gateway 路由目标推导执行模式。

    Args:
        route_target: gateway 产出的目标节点名。

    Returns:
        str: 模式字符串：
            - `fast_lane`: 订单/商品直达；
            - `chat`: 闲聊节点；
            - `supervisor_loop`: 进入动态主管慢车道。
    """

    if route_target in {"order_agent", "product_agent"}:
        return "fast_lane"
    if route_target == "chat_agent":
        return "chat"
    return "supervisor_loop"


@traceable(name="Supervisor Gateway Router Node", run_type="chain")
def gateway_router(state: AgentState) -> dict[str, Any]:
    """
    执行前置意图网关路由，并同步下发任务难度与模型策略。

    Args:
        state: 当前图状态，主要读取 `user_input` 与 `messages`（用于轻量历史辅助判断）。

    Returns:
        dict[str, Any]: gateway 的状态更新字典，字段说明如下：
            - `routing.route_target`: 路由目标节点；
            - `routing.mode`: 执行模式（fast_lane/chat/supervisor_loop）；
            - `routing.task_difficulty`: 任务难度（simple/normal/complex）；
            - `routing.selected_model`: 下游建议模型名；
            - `routing.think_enabled`: 是否开启深度思考；
            - `next_node`: 与 route_target 保持一致；
            - `context`: 初始化共享上下文（含调用计数、agent_outputs、原始输入）；
            - `execution_traces`: 路由决策追踪。
    """

    user_input = str(state.get("user_input") or "").strip()
    history_messages = list(state.get("messages") or [])

    input_messages: list[Any] = [SystemMessage(content=_GATEWAY_PROMPT)]
    if history_messages:
        input_messages.extend(history_messages[-8:])
    elif user_input:
        input_messages.append(HumanMessage(content=user_input))

    route_target = "supervisor_agent"
    task_difficulty = NORMAL_DIFFICULTY

    gateway_model_name = "qwen-flash"
    try:
        llm = create_chat_model(
            model=gateway_model_name,
            temperature=0,
            response_format={"type": "json_object"},
        )
        response = llm.invoke(input_messages)
        payload = json.loads(str(response.content))
        route_target, task_difficulty = build_gateway_decision(payload)
    except Exception:
        route_target, task_difficulty = "supervisor_agent", NORMAL_DIFFICULTY

    mode = _resolve_mode(route_target)
    profile = resolve_model_profile(task_difficulty)
    routing = apply_model_profile_to_routing(
        {
            "route_target": route_target,
            "mode": mode,
            "turn": 0,
            "finished": False,
        },
        task_difficulty=task_difficulty,
        profile=profile,
    )

    update: dict[str, Any] = {
        "routing": routing,
        "next_node": route_target,
        "context": {
            "node_call_counts": {},
            "agent_outputs": {},
            "original_user_input": user_input,
        },
    }
    update.update(
        build_execution_trace_update(
            node_name="gateway_router",
            model_name=gateway_model_name,
            input_messages=serialize_messages(input_messages),
            output_text=json.dumps(update, ensure_ascii=False, default=str),
            tool_calls=[],
        )
    )
    return update
