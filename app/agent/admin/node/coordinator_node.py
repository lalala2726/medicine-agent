import json
from typing import Any

from langchain_core.messages import HumanMessage

from app.agent.admin.agent_state import AgentState, PlanStep
from app.agent.admin.agent_utils import build_execution_trace_update
from app.agent.admin.dag_rules import (
    build_retry_feedback,
    review_plan,
    select_model_by_difficulty,
    should_enable_thinking_by_difficulty,
)
from app.agent.admin.history_utils import build_messages_with_history
from app.agent.tools.coordinator_tools import get_agent_detail
from app.core.assistant_status import status_node
from app.core.langsmith import traceable
from app.core.llm import create_chat_model
from app.utils.streaming_utils import invoke_with_policy

_system_prompt = """
    你是后台工作流中的 **coordinator_agent**，负责将用户请求拆解成**极简、高效、无冗余的 DAG 计划**（仅规划，不执行）。
    
    **核心目标**：  
    让最终呈现给用户的输出**尽可能直接、清晰、自然**，杜绝任何不必要的中间节点（尤其是 summary_agent）。
    
    ### 可用执行节点（node_name 只能从以下值中选择）
    - order_agent: 订单查询、订单状态与订单信息核验。
    - product_agent: 商品查询、商品详情查询。
    - excel_agent: 表格解析、表格整理、Excel 导出。
    - chart_agent: 基于已有结构化数据生成图表或统计说明。
    - summary_agent: 对多个节点结果做汇总并输出最终结论（**仅在必要时使用**）。
    
    **关键决策规则（必须严格遵守）—— 何时使用 summary_agent**
    
    **绝不使用 summary_agent 的场景**（直接让最后一个业务节点 final_output=true）：
    - 用户只想**获取并展示**具体业务数据（如“最近1个订单的商品信息”“订单详情”“商品价格”等）。
    - 任务是**纯查询 + 链式获取**（order → product、product → 其他等）。
    - 示例：用户说“最近1个订单的商品信息” → 正确计划：
      - s1: order_agent（只返回最新1个订单的 product_id）
      - s2: product_agent（final_output=true）→ 直接输出商品完整详情（名称、价格、规格、图片等），**无需再经过 summary**。
    
    **必须使用 summary_agent 的场景**：
    - 用户明确要求“总结”“汇总”“分析”“对比”“报告”“解释原因”等。
    - 有同时并行节点，或需要跨多个结果做聚合/统计/图表说明。
    - 需要生成自然语言长文、结论、建议等。
    
    **最小必要输出原则（强化版）**
    - 上游节点**永远只输出下游所需的最小字段**（通常是 ID、order_no、product_id 等）。
    - task_description 示例（必须严格按此风格）：
    
      ```text
      查询用户最近1个订单，只返回该订单的 product_id和订单编号，除此之外不要输出任何订单详情、时间、状态等其他字段。
      ```
    
      ```text
      接收上游提供的 product_id 列表，查询每个商品的完整详情（名称、价格、规格、库存、图片链接等），以清晰的结构化 JSON 输出，直接满足用户查看需求，无需额外总结。
      ```
    
    **输出要求**：  
    严格只输出以下 JSON，不要包含任何其他文字、解释、代码块标记：
    
    ```json
    {
      "plan": [
        {
          "step_id": "s1",
          "node_name": "order_agent",
          "task_description": "查询用户最近1个订单，只返回该订单的 product_id和订单号JSON方式输出",
          "required_depends_on": [],
          "optional_depends_on": [],
          "read_from": [],
          "include_user_input": false,
          "include_chat_history": false,
          "final_output": false,
          "failure_policy": { ... }
        },
        {
          "step_id": "s2",
          "node_name": "product_agent",
          "task_description": "根据上游提供的 product_id，查询商品完整详情整理好使用markdown发送给用户",
          "required_depends_on": ["s1"],
          "optional_depends_on": [],
          "read_from": ["s1"],
          "include_user_input": false,
          "include_chat_history": false,
          "final_output": true
        }
      ]
    }
    
    严格规则：
    1. 顶层只允许一个键：plan，且 plan 必须是对象数组。
    2. 每个步骤必须包含：step_id/node_name/task_description/required_depends_on/optional_depends_on/read_from/include_user_input/include_chat_history/final_output。
    3. step_id 必须全局唯一。
    4. node_name 不得为 coordinator_agent，且必须是可用节点之一。
    5. required_depends_on / optional_depends_on / read_from 必须是字符串数组，且引用的 step_id 必须存在。
    6. required_depends_on 与 optional_depends_on 不能重复引用同一上游步骤。
    7. read_from 只能读取该步骤“可达上游”步骤，不能读取无依赖关系步骤，也不能读取自身。
    8. plan 必须无环。
    9. final_output=true 必须且仅能有一个。
    10. final_output 对应步骤不能被其他步骤依赖（必须是最终输出端）。
    11. failure_policy 为可选字段；若提供，必须包含合法值：
        - mode: hybrid|marker_only|tool_only
        - error_marker_prefix: 非空字符串（推荐 "__ERROR__:"）
        - tool_error_counting: consecutive|total
        - max_tool_errors: 1..5（推荐 2）
        - strict_data_quality: true|false（推荐 true）
    12. 若需求不清晰，输出最小可执行 DAG（至少 1 个步骤，且 final_output 唯一）。
"""

_PLAN_REVIEW_RETRY_LIMIT = 2


def _serialize_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """
    序列化节点输入消息，供 execution_trace 落库。

    Args:
        messages: 原始消息列表。

    Returns:
        list[dict[str, Any]]: 序列化后的消息字典列表。
    """
    serialized: list[dict[str, Any]] = []
    for message in messages:
        serialized.append(
            {
                "role": str(getattr(message, "type", "") or message.__class__.__name__).strip().lower() or "unknown",
                "content": getattr(message, "content", ""),
            }
        )
    return serialized


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
    enable_thinking = should_enable_thinking_by_difficulty(difficulty)

    user_input = str(state.get("user_input") or "").strip()
    if not user_input:
        routing["difficulty"] = difficulty
        result = {"routing": routing, "plan": []}
        result.update(
            build_execution_trace_update(
                node_name="coordinator_agent",
                model_name="unknown",
                input_messages=[],
                output_text=json.dumps(result, ensure_ascii=False, default=str),
                tool_calls=[],
            )
        )
        return result
    history_messages = list(state.get("history_messages") or [])

    llm = create_chat_model(
        model=model_name,
        temperature=0,
        response_format={"type": "json_object"},
        extra_body={"enable_thinking": True} if enable_thinking else None,
    )
    base_messages = build_messages_with_history(
        system_prompt=_system_prompt,
        history_messages=history_messages,
        fallback_user_input=user_input,
    )

    routing["difficulty"] = difficulty
    reviewed_plan: list[PlanStep] = []
    last_reason = "模型返回内容不是有效 JSON。"
    last_messages_for_trace = list(base_messages)
    last_tool_calls: list[dict[str, Any]] = []

    for attempt in range(_PLAN_REVIEW_RETRY_LIMIT + 1):
        messages = list(base_messages)
        if attempt > 0:
            messages.append(HumanMessage(content=build_retry_feedback(last_reason)))
        last_messages_for_trace = list(messages)

        try:
            payload, tool_calls = _invoke_coordinator_payload(llm, messages)
            last_tool_calls = list(tool_calls)
        except Exception:
            last_reason = "模型返回内容不是有效 JSON。"
            continue

        is_valid, candidate_plan, reason = review_plan(payload.get("plan"), difficulty)
        if is_valid:
            reviewed_plan = candidate_plan
            break
        last_reason = reason

    result = {
        "routing": routing,
        "plan": reviewed_plan,
    }
    result.update(
        build_execution_trace_update(
            node_name="coordinator_agent",
            model_name=model_name,
            input_messages=_serialize_messages(last_messages_for_trace),
            output_text=json.dumps(result, ensure_ascii=False, default=str),
            tool_calls=last_tool_calls,
        )
    )
    return result


def _invoke_coordinator_payload(llm: Any, messages: list[Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    调用 coordinator 模型并解析 JSON 结果。

    优先使用工具模式（支持 bind_tools 的模型），让模型可按需调用 get_agent_detail；
    若模型不支持工具绑定，则回退为普通 invoke 调用。

    Args:
        llm: coordinator 模型实例。
        messages: 输入消息列表。

    Returns:
        tuple[dict[str, Any], list[dict[str, Any]]]:
            - payload: 解析后的 JSON 计划结果
            - tool_calls: 工具调用明细（用于 execution_trace）
    """
    bind_tools = getattr(llm, "bind_tools", None)
    if callable(bind_tools):
        content, diagnostics = invoke_with_policy(
            llm,
            messages,
            tools=[get_agent_detail],
            enable_stream=False,
        )
        return json.loads(str(content)), list(diagnostics.get("tool_call_details") or [])

    response = llm.invoke(messages)
    return json.loads(str(response.content)), []
