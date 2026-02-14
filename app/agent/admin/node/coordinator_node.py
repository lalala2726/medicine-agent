import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.admin.agent_state import AgentState, PlanStep
from app.agent.admin.dag_rules import (
    build_retry_feedback,
    review_plan,
    select_model_by_difficulty,
)
from app.agent.tools.coordinator_tools import get_agent_detail
from app.core.assistant_status import status_node
from app.core.langsmith import traceable
from app.core.llm import create_chat_model
from app.utils.streaming_utils import invoke

_system_prompt = """
    你是后台工作流中的 coordinator_agent，负责把用户请求拆解成 DAG 计划（仅规划，不执行）。

    可用执行节点（node_name 只能从以下值中选择）：
    - order_agent: 订单查询、订单状态与订单信息核验。
    - excel_agent: 表格解析、表格整理、Excel 导出。
    - chart_agent: 基于已有结构化数据生成图表或统计说明。
    - summary_agent: 对多个节点结果做汇总并输出最终结论。
    - product_agent: 商品查询、商品详情查询。

    上述只是简单描述，如果你需要调用某些节点，你必须调用 get_agent_detail 获取节点详细功能介绍。
    工具调用参数：
    - agent_names: string[]，节点名列表（例如 ["order_agent", "product_agent"]，也支持 ["all"]）
    - include_tool_parameters: bool，是否返回该节点可用工具和工具参数说明（默认 true）
    - include_coordination_guide: bool，是否返回该节点协同建议（默认 true）
    - include_plan_examples: bool，是否返回该节点计划片段示例（默认 false）
    建议按需查询，不要一次性拉取全部节点细节，以节省上下文。

    注意：
    - chat_agent 由 gateway_router 处理，禁止出现在 plan 中。
    - 你必须输出新格式，不允许旧的分阶段嵌套数组结构。

    输出 JSON 格式（仅输出 JSON，不要额外文本）：
    {
      "plan": [
        {
          "step_id": "s1",
          "node_name": "order_agent",
          "task_description": "string",
          "depends_on": [],
          "read_from": [],
          "include_user_input": false,
          "include_chat_history": false,
          "final_output": false
        },
        {
          "step_id": "s2",
          "node_name": "summary_agent",
          "task_description": "string",
          "depends_on": ["s1"],
          "read_from": ["s1"],
          "include_user_input": false,
          "include_chat_history": false,
          "final_output": true
        }
      ]
    }

    严格规则：
    1. 顶层只允许一个键：plan，且 plan 必须是对象数组。
    2. 每个步骤必须包含：step_id/node_name/task_description/depends_on/read_from/include_user_input/include_chat_history/final_output。
    3. step_id 必须全局唯一。
    4. node_name 不得为 coordinator_agent，且必须是可用节点之一。
    5. depends_on / read_from 必须是字符串数组，且引用的 step_id 必须存在。
    6. read_from 只能读取该步骤“可达上游”步骤，不能读取无依赖关系步骤，也不能读取自身。
    7. plan 必须无环。
    8. final_output=true 必须且仅能有一个。
    9. final_output 对应步骤不能被其他步骤依赖（必须是最终输出端）。
    10. 若需求不清晰，输出最小可执行 DAG（至少 1 个步骤，且 final_output 唯一）。
"""

_PLAN_REVIEW_RETRY_LIMIT = 2


@status_node(node="coordinator", start_message="正在规划任务中")
@traceable(name="Coordinator Agent Node", run_type="chain")
def coordinator(state: AgentState) -> dict[str, Any]:
    """
    生成并审核 DAG 计划的协调器入口函数。

    职责边界：
    - 负责组装提示词、调用 LLM 生成候选计划
    - 通过 `dag_rules.review_plan` 审核计划
    - 在审核失败时基于 `dag_rules.build_retry_feedback` 触发有限次重试
    - 不承载审核规则细节与调度规则细节

    Args:
        state: 当前 Agent 状态，主要读取：
            - `state.user_input`: 用户请求文本
            - `state.routing.difficulty`: 路由阶段判定的复杂度

    Returns:
        包含以下字段的增量更新：
            - `routing`: 至少包含规范化后的 `difficulty`
            - `plan`: 审核通过的规范化计划；若无输入或重试后仍失败则为空列表
    """
    routing = dict(state.get("routing") or {})
    difficulty = str(routing.get("difficulty") or "medium").strip().lower()
    model_name = select_model_by_difficulty(difficulty)

    user_input = str(state.get("user_input") or "").strip()
    if not user_input:
        routing["difficulty"] = difficulty
        return {"routing": routing, "plan": []}

    llm = create_chat_model(
        model=model_name,
        temperature=0,
        response_format={"type": "json_object"},
    )
    base_messages = [
        SystemMessage(content=_system_prompt),
        HumanMessage(content=f"用户请求：{user_input}"),
    ]

    routing["difficulty"] = difficulty
    reviewed_plan: list[PlanStep] = []
    last_reason = "模型返回内容不是有效 JSON。"

    for attempt in range(_PLAN_REVIEW_RETRY_LIMIT + 1):
        messages = list(base_messages)
        if attempt > 0:
            messages.append(HumanMessage(content=build_retry_feedback(last_reason)))

        try:
            payload = _invoke_coordinator_payload(llm, messages)
        except Exception:
            last_reason = "模型返回内容不是有效 JSON。"
            continue

        is_valid, candidate_plan, reason = review_plan(payload.get("plan"), difficulty)
        if is_valid:
            reviewed_plan = candidate_plan
            break
        last_reason = reason

    return {
        "routing": routing,
        "plan": reviewed_plan,
    }


def _invoke_coordinator_payload(llm: Any, messages: list[Any]) -> dict[str, Any]:
    """
    调用 coordinator 模型并解析 JSON 结果。

    优先使用工具模式（支持 bind_tools 的模型），让模型可按需调用 get_agent_detail；
    若模型不支持工具绑定，则回退为普通 invoke 调用。
    """
    bind_tools = getattr(llm, "bind_tools", None)
    if callable(bind_tools):
        content = invoke(llm, messages, tools=[get_agent_detail])
        return json.loads(str(content))

    response = llm.invoke(messages)
    return json.loads(str(response.content))
