from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.admin.agent_utils import build_execution_trace_update, serialize_messages
from app.agent.admin.model_policy import (
    DEFAULT_NODE_GOAL,
    NORMAL_DIFFICULTY,
    apply_model_profile_to_routing,
    build_supervisor_decision,
    normalize_task_difficulty,
    resolve_model_profile,
)
from app.agent.admin.state import AgentState
from app.core.assistant_status import emit_sse_response, has_status_emitter
from app.core.langsmith import traceable
from app.core.llm import create_chat_model
from app.schemas.sse_response import AssistantResponse, Content, MessageType

# 同一目标节点在单轮任务中的最大调度次数，超过后强制切到 summary 防止死循环。
_MAX_NODE_CALLS = 2

_SUPERVISOR_PROMPT = """
        你是药品商城后台管理系统的动态主管节点（supervisor_agent）。
        你的职责只有一个：基于当前上下文决定“下一步调用哪个业务节点”。
        
        你不是最终回答节点，不直接对用户输出答案，不做流式回复。
        你必须只输出一个 JSON 对象，且格式必须严格为：
        {"target_node":"order_agent|product_agent|excel_agent|summary_agent","task_difficulty":"simple|normal|complex","node_goal":"该节点本步目标"}
        
        【节点能力说明（按工具能力）】
        1) order_agent（订单域）
        - 可调用工具：get_order_list, get_orders_detail
        - 适用：查订单列表、订单详情、收货人信息、从订单明细提取 product_id。
        
        2) product_agent（商品域）
        - 可调用工具：get_product_list, get_product_detail, get_drug_detail
        - 适用：查商品详情、库存/上下架状态、药品说明书与药品详情。
        
        3) excel_agent（表格域）
        - 当前为占位能力（尚未实现完整工具链）。
        - 仅在明确表格处理任务且其他节点无法完成时调度。
        
        4) summary_agent（最终汇总）
        - 当你判断任务已可收敛时，直接输出 target_node=summary_agent。
        - node_goal 必须写清用户最关心的信息和整理方式（例如：表格、按订单分组、先结论后明细）。
        
        【硬约束】
        1. 只输出 JSON，不得输出解释文本、Markdown、代码块。
        2. 优先复用 context.agent_outputs 中已有结构化信息，禁止重复索要用户已给参数。
        3. target_node 只能是 order_agent/product_agent/excel_agent/summary_agent。
        4. task_difficulty 只能是 simple|normal|complex，不确定时输出 normal。
        5. node_goal 必须非空，且要可执行、可落地，避免空泛描述。
        6. order_agent 与 product_agent 不直接互读对方输出；你必须把需要传递的信息写进 node_goal。
        7. 若是跨域任务，优先先订单后商品；当已收集足够信息时再调度 summary_agent。
        
        【示例：用户要前5个订单的商品说明书】
        示例输出1（先查订单并提取商品ID）：
        {"target_node":"order_agent","task_difficulty":"complex","node_goal":"查询最近5个订单，只输出订单编号与商品ID映射，减少无关字段输出。"}
        
        示例输出2（基于商品ID查说明书）：
        {"target_node":"product_agent","task_difficulty":"normal","node_goal":"使用 order_agent 已输出的商品ID（P1,P2,P3）批量查询药品说明书，返回每个商品的说明书要点。"}
        
        示例输出3（交给最终汇总）：
        {"target_node":"summary_agent","task_difficulty":"normal","node_goal":"围绕用户关心的商品说明书给出结论，先给总体结论，再按订单分组用表格展示订单号、商品ID、说明书要点；缺失项单独说明原因和下一步建议。"}
"""


def _emit_supervisor_enter_notice(routing: dict[str, Any]) -> None:
    """
    发射“进入 Supervisor 决策轮”的状态通知。

    Args:
        routing: 当前路由元信息字典，主要用于补充 `turn` 与 `task_difficulty`
            到事件元数据，便于前端和日志侧定位当前决策轮次。

    Returns:
        None: 该函数只做 SSE NOTICE 事件发射，不返回业务数据。
    """

    if not has_status_emitter():
        return

    emit_sse_response(
        AssistantResponse(
            content=Content(
                node="supervisor_agent",
                state="start",
                message="进入 Supervisor 动态决策",
            ),
            type=MessageType.NOTICE,
            meta={
                "turn": int(routing.get("turn") or 0),
                "task_difficulty": routing.get("task_difficulty"),
            },
        )
    )


def _emit_supervisor_dispatch_notice(
    *,
    target_node: str,
    routing: dict[str, Any],
) -> None:
    """
    发射 Supervisor 下一跳通知（不输出 node_goal 详情，避免前端重复冗长文案）。

    Args:
        target_node: Supervisor 本轮决策的下一跳目标节点。
        routing: 已更新完成的路由字典，用于附带难度与模型策略元信息。

    Returns:
        None: 该函数仅负责发射 NOTICE 事件，不参与状态计算。
    """

    if not has_status_emitter():
        return

    message = (
        "Supervisor 判断任务完成，准备交给 summary_agent 汇总输出"
        if target_node == "summary_agent"
        else f"Supervisor 下一跳节点: {target_node}"
    )
    emit_sse_response(
        AssistantResponse(
            content=Content(
                node="supervisor_agent",
                state="dispatch",
                message=message,
                name=target_node,
            ),
            type=MessageType.NOTICE,
            meta={
                "target_node": target_node,
                "task_difficulty": routing.get("task_difficulty"),
                "selected_model": routing.get("selected_model"),
                "think_enabled": routing.get("think_enabled"),
                "turn": routing.get("turn"),
            },
        )
    )


@traceable(name="Supervisor Agent Node", run_type="chain")
def supervisor_agent(state: AgentState) -> dict[str, Any]:
    """
    执行 Supervisor 单步决策，并把下一跳写回共享状态。

    行为边界：
    1. 只做“下一步调用谁”的决策，不直接向用户回答业务结果；
    2. 只输出受控 JSON 契约字段（经模型输出解析后写入状态）；
    3. 保留同节点调用次数限制，避免慢车道无限回环。

    Args:
        state: 当前图状态，核心读取字段如下：
            - `messages`: 全量历史消息（用于记忆与上下文续接）。
            - `context`: 共享上下文（提取 ID、历史工具产出、调用计数等）。
            - `routing`: 路由元信息（mode、turn、task_difficulty 等）。

    Returns:
        dict[str, Any]: Supervisor 状态增量，关键字段如下：
            - `routing.target_node`: 下一跳业务节点（含 `summary_agent`）。
            - `routing.node_goal`: 下游节点本步执行目标。
            - `routing`: 更新后的路由信息（含 `finished`、模型策略字段）。
            - `context.node_call_counts`: 节点调用计数更新结果。
            - `execution_traces`: 本轮模型输入与决策结果追踪。
    """

    context = dict(state.get("context") or {})
    routing = dict(state.get("routing") or {})
    counts = dict(context.get("node_call_counts") or {})
    all_messages = list(state.get("messages") or [])

    current_difficulty = normalize_task_difficulty(
        routing.get("task_difficulty") or NORMAL_DIFFICULTY
    )
    _emit_supervisor_enter_notice(routing)
    current_profile = resolve_model_profile(current_difficulty)

    supervisor_input = {
        "user_input": state.get("user_input"),
        "context": context,
        "routing": routing,
        "messages": serialize_messages(all_messages),
    }
    input_messages: list[Any] = [
        SystemMessage(content=_SUPERVISOR_PROMPT),
        HumanMessage(content=json.dumps(supervisor_input, ensure_ascii=False, default=str)),
    ]

    supervisor_model_name = str(current_profile.get("model") or "qwen-plus")
    supervisor_think_enabled = bool(current_profile.get("think"))
    try:
        llm = create_chat_model(
            model=supervisor_model_name,
            think=supervisor_think_enabled,
            temperature=0,
            response_format={"type": "json_object"},
        )
        response = llm.invoke(input_messages)
        payload = json.loads(str(response.content))
        target_node, node_goal, task_difficulty = build_supervisor_decision(
            payload,
            fallback_task_difficulty=current_difficulty,
        )
    except Exception:
        target_node, node_goal, task_difficulty = "summary_agent", DEFAULT_NODE_GOAL, current_difficulty

    if target_node != "summary_agent" and int(counts.get(target_node, 0)) >= _MAX_NODE_CALLS:
        target_node = "summary_agent"
        node_goal = DEFAULT_NODE_GOAL

    if target_node != "summary_agent":
        counts[target_node] = int(counts.get(target_node, 0)) + 1
    context["node_call_counts"] = counts

    resolved_profile = resolve_model_profile(task_difficulty)
    routing = apply_model_profile_to_routing(
        routing,
        task_difficulty=task_difficulty,
        profile=resolved_profile,
    )
    routing["turn"] = int(routing.get("turn") or 0) + 1
    routing["target_node"] = target_node
    routing["finished"] = target_node == "summary_agent"
    routing["node_goal"] = str(node_goal or "").strip() or DEFAULT_NODE_GOAL
    routing["route_target"] = target_node

    _emit_supervisor_dispatch_notice(
        target_node=target_node,
        routing=routing,
    )

    update: dict[str, Any] = {
        "routing": routing,
        "context": context,
    }
    update.update(
        build_execution_trace_update(
            node_name="supervisor_agent",
            model_name=supervisor_model_name,
            input_messages=serialize_messages(input_messages),
            output_text=json.dumps(update, ensure_ascii=False, default=str),
            tool_calls=[],
        )
    )
    return update
