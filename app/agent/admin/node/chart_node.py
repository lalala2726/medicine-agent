from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.admin.agent_utils import (
    build_standard_node_update,
    execute_tool_node,
)
from app.agent.admin.agent_state import AgentState
from app.agent.admin.node.runtime_context import (
    build_step_runtime,
)
from app.agent.tools.chart_tools import CHART_TOOLS
from app.core.assistant_status import status_node
from app.core.langsmith import traceable
from app.core.llm import create_chat_model
from app.schemas.prompt import base_prompt
from app.utils.streaming_utils import is_final_node

_CHART_SYSTEM_PROMPT = (
        """
        你是后台工作流中的 chart_agent，职责是基于已有结构化数据生成可渲染图表配置。
        
        严格规则：
        1. 先看输入上下文里的真实数据，不编造数据。
        2. 只能调用 get_chart_sample_by_name 获取图表模板，不需要调用其他图表类型工具。
        3. 先根据下方图表类型清单选择最匹配的 type，再调用 get_chart_sample_by_name(name_or_type=<type>)。
        4. get_chart_sample_by_name 的返回值就是图表示例模板本体，禁止依赖额外元数据字段。
        5. 输出时必须保持模板字段结构，仅替换 data/categories/series 等数据值。
        6. 如果没有足够数据生成图表，明确返回“暂无相关数据，无法生成图表”。
        
        支持图表类型，在使用 get_chart_sample_by_name 时，必须指定 type 参数，格式为： 如 line
        - line（折线图）：时间序列趋势对比（单条或多分组）
        - column（柱状图）：类别对比、排名（纵向或横向）
        - bar（条形图）：水平排名/对比
        - pie（饼图）：占比分布、部分与整体关系
        - area（面积图）：趋势展示且强调量级
        - scatter（散点图）：两个变量相关性
        - histogram（直方图）：数值分布/频次统计
        - dualaxes（双轴图）：两个量纲对比（混合折线/柱）
        - funnel（漏斗图）：流程转化与流失分析
        - radar（雷达图）：多维度能力/评分对比
        - treemap（矩阵树图）：层级占比对比
        - wordcloud（词云图）：关键词频率/权重
        - mindmap（思维导图）：层级/脑图展示
        - networkgraph（网络图）：关系网络、人物关系
        - flowdiagram（流程图）：业务流程/决策路径
        - organizationchart（组织架构图）：组织结构展示
        - indentedtree（缩进树）：目录/知识树结构
        - fishbonediagram（鱼骨图）：因果/根因分析
    """
        + base_prompt
)


def _build_chart_input(state: AgentState, runtime: dict[str, Any]) -> dict[str, Any]:
    """
    构建 chart 节点输入载荷。

    Args:
        state: 当前全局状态，主要读取 `errors`。
        runtime: `build_step_runtime` 输出，包含任务描述、上游输出与上下文开关。

    Returns:
        dict[str, Any]: 图表节点可直接序列化给模型的输入结构。
    """
    # 图表节点仅消费任务描述、历史/用户输入开关、上游输出和错误信息。
    payload: dict[str, Any] = {
        "task_description": runtime.get("task_description") or "根据已有数据生成图表",
        "upstream_outputs": runtime.get("upstream_outputs") or {},
        "read_from": runtime.get("read_from") or [],
        "errors": state.get("errors") or [],
    }
    user_input = runtime.get("user_input")
    if isinstance(user_input, str) and user_input:
        payload["user_input"] = user_input

    history_messages = runtime.get("history_messages_serialized")
    if isinstance(history_messages, list) and history_messages:
        payload["history_messages"] = history_messages

    failure_policy = runtime.get("failure_policy") or {}
    if failure_policy.get("strict_data_quality", True):
        payload["failure_policy_hint"] = (
            "当工具调用连续失败或数据不可信/不完整时，必须以 '__ERROR__:' 前缀输出错误原因。"
        )
    return payload


@status_node(
    node="chart",
    start_message="正在分析数据准备生成图表",
    display_when="after_coordinator",
)
@traceable(name="Chart Agent Node", run_type="chain")
def chart_agent(state: AgentState) -> dict:
    """
    执行图表节点，基于上游结构化数据生成图表配置。

    Args:
        state: 当前全局状态，包含步骤配置与上游输出。

    Returns:
        dict: 节点增量更新（results、step_outputs、execution_traces）。
    """
    # 读取当前步骤配置（read_from、final_output、上下文开关）。
    runtime = build_step_runtime(
        state,
        "chart_agent",
        default_task_description="根据已有数据生成图表",
    )
    chart_input = _build_chart_input(state, runtime)
    failure_policy = runtime.get("failure_policy") or {}
    final_output = is_final_node(state, "chart_agent")
    llm = create_chat_model(model="qwen-plus", temperature=0.1)
    messages = [
        SystemMessage(content=_CHART_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(chart_input, ensure_ascii=False)),
    ]
    execution_result = execute_tool_node(
        llm=llm,
        messages=messages,
        tools=CHART_TOOLS,
        enable_stream=final_output,
        failure_policy=failure_policy,
        fallback_content="图表服务暂时不可用，请稍后重试。",
        fallback_error="chart_agent 执行失败",
    )
    return build_standard_node_update(
        state=state,
        runtime=runtime,
        node_name="chart_agent",
        result_key="chart",
        execution_result=execution_result,
        is_end=final_output,
        step_output_payload={
            "chart": {
                "content": execution_result.content,
                "is_end": final_output,
            }
        },
    )
