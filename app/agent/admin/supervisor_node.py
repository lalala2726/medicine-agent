import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.admin.agent_state import AgentState
from app.core.llm import create_chat_model

_system_prompt = """
     你是运行在药品商城后台的 AI 管理助手，负责对用户请求进行统一决策与编排。

    你的核心职责包括：
    1. 准确理解用户的自然语言请求；
    2. 判断请求的复杂度，决定是否需要调用一个或多个专业 Agent；
    3. 对复杂请求进行任务拆解，生成清晰的执行计划，并将子任务分配给合适的 Agent；
    4. 在所有 Agent 完成任务后，对结果进行整理、汇总，并生成最终回复返回给用户。
    
    工作规则：
    - 如果用户请求简单、无需跨业务域或多步骤处理，你可以直接生成并返回结果；
    - 如果用户请求涉及多个业务域、需要查询、处理或组合数据，你必须先进行任务规划，再调用相应的 Agent 执行；
    - 你只负责任务规划、调度和结果汇总，不直接执行任何具体业务操作。
    
    当前可用的 Agent 包括：
    - 订单 Agent：负责订单相关任务，如订单查询、订单状态处理等；
    - 售后 Agent：负责售后相关任务，如售后查询、售后状态处理等；
    - Excel Agent：负责解析和处理 Excel 文件，如解析一个Excel URL ，或者是将一个信息整理成Excel文件然后返回一个下载链接
    - Summary agent：负责总结、汇总和生成最终结果。通常是两个并行节点汇总到本节点之后进行汇总
    - Coordinator Agent：这是你当前的 Agent，负责任务规划、生成计划并统一调度上面提到的 Agent。
    
    Agent 调用规则：
    - 你可以同时调用多个 Agent 并行执行任务；
    - 在返回最终结果前，必须等待所有被调用的 Agent 执行完成，并统一汇总结果。
    
    当任务出现复杂的时候
    
    数据安全规则：
    - 涉及数据修改的操作，原则上需要用户确认；
    - 当前处于测试阶段，暂不要求用户进行确认。
    
    你的目标是：
    在保证结果准确、逻辑清晰的前提下，尽量减少用户交互成本，高效地向用户提供系统信息。

    你必须输出 JSON，并且只输出 JSON：
    {
      "user_intent": {"type": "string", "summary": "string"},
      "plan": [
        {"node_name": "order_agent|excel_agent|chart_agent", "task_description": "string", "last_node": ["coordinator_agent"]},
        [
          {"node_name": "order_agent", "task_description": "string", "last_node": ["coordinator_agent"]},
          {"node_name": "excel_agent", "task_description": "string", "last_node": ["coordinator_agent"]}
        ]
      ]
    }

    输出规则：
    1. 能并行的步骤请放到同一个数组中。
    2. node_name 只能使用 order_agent / excel_agent / chart_agent。
    3. 如果无法明确拆解，也要给出最小可执行 plan。
    """


_COORDINATOR_MODEL_BY_DIFFICULTY = {
    "simple": "qwen-flash",
    "medium": "qwen-plus",
    "complex": "qwen-max",
}


def _select_model_by_difficulty(difficulty: str) -> str:
    key = str(difficulty or "simple").strip().lower()
    return _COORDINATOR_MODEL_BY_DIFFICULTY.get(key, _COORDINATOR_MODEL_BY_DIFFICULTY["simple"])


def coordinator(state: AgentState) -> dict[str,Any]:
    routing = dict(state.get("routing") or {})
    difficulty = str(routing.get("difficulty") or "medium").strip().lower()
    model_name = _select_model_by_difficulty(difficulty)

    user_input = str(state.get("user_input") or "").strip()
    if not user_input:
        routing["difficulty"] = difficulty
        return {"routing": routing, "user_intent": {}, "plan": []}

    llm = create_chat_model(
        model=model_name,
        temperature=0,
        response_format={"type": "json_object"},
    )
    messages = [
        SystemMessage(content=_system_prompt),
        HumanMessage(content=f"用户请求：{user_input}"),
    ]
    try:
        response = llm.invoke(messages)
        payload = json.loads(str(response.content))
    except Exception:
        routing["difficulty"] = difficulty
        return {"routing": routing, "user_intent": {}, "plan": []}

    routing["difficulty"] = difficulty
    user_intent = payload.get("user_intent")
    if not isinstance(user_intent, dict):
        user_intent = {}

    plan = payload.get("plan")
    if not isinstance(plan, list):
        plan = []

    return {
        "routing": routing,
        "user_intent": user_intent,
        "plan": plan,
    }
