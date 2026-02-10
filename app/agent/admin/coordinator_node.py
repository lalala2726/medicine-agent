import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.admin.agent_state import AgentState
from app.core.langsmith import traceable
from app.core.llm import create_chat_model

_system_prompt = """
你是后台工作流中的 coordinator_agent，负责把用户请求拆解成可执行计划。

你的职责：
1. 根据任务依赖关系生成 plan（支持串行与并行阶段）。
2. 只做规划，不执行具体业务。

可用执行节点（node_name 只能从以下值中选择）：
- order_agent: 订单查询、订单状态与订单信息核验。
- excel_agent: 表格解析、表格整理、Excel 导出。
- chart_agent: 基于已有结构化数据生成图表或统计说明。
- summary_agent: 对多个节点结果做汇总并输出最终结论。

注意：
- chat_agent 由 gateway_router 处理，不要出现在 plan 中。

你输出的 JSON 必须与状态结构兼容（对应 AgentState 中的 plan）：
{
  "plan": [
    {
      "node_name": "order_agent",
      "task_description": "string",
      "last_node": ["coordinator_agent"]
    },
    [
      {
        "node_name": "excel_agent",
        "task_description": "string",
        "last_node": ["order_agent"]
      },
      {
        "node_name": "chart_agent",
        "task_description": "string",
        "last_node": ["order_agent"]
      }
    ]
  ]
}

严格规则：
1. 只输出 JSON，不要输出任何解释、Markdown、代码块标记。
2. 顶层只允许一个键：plan。
3. plan 必须是数组。
4. plan 每个元素要么是单个步骤对象（串行），要么是步骤对象数组（并行阶段）。
5. 每个步骤对象必须包含：node_name、task_description、last_node。
7. last_node 必须是字符串数组：
   - 第一阶段通常为 ["coordinator_agent"]。
   - 后续阶段填写其依赖的上游节点名数组。
8. 不要产生循环依赖，不要把 coordinator_agent 作为 plan 的 node_name。
9. 若用户需求不清晰，给出最小可执行 plan（至少 1 个步骤）。
"""

_COORDINATOR_MODEL_BY_DIFFICULTY = {
    "simple": "qwen-flash",
    "medium": "qwen-plus",
    "complex": "qwen-max",
}


def _select_model_by_difficulty(difficulty: str) -> str:
    key = str(difficulty or "simple").strip().lower()
    return _COORDINATOR_MODEL_BY_DIFFICULTY.get(key, _COORDINATOR_MODEL_BY_DIFFICULTY["simple"])


@traceable(name="Coordinator Agent Node", run_type="chain")
def coordinator(state: AgentState) -> dict[str, Any]:
    routing = dict(state.get("routing") or {})
    difficulty = str(routing.get("difficulty") or "medium").strip().lower()
    model_name = _select_model_by_difficulty(difficulty)

    user_input = str(state.get("user_input") or "").strip()
    if not user_input:
        routing["difficulty"] = difficulty
        return {"routing": routing, "plan": []}

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
        return {"routing": routing, "plan": []}

    routing["difficulty"] = difficulty
    plan = payload.get("plan")
    if not isinstance(plan, list):
        plan = []

    return {
        "routing": routing,
        "plan": plan,
    }
