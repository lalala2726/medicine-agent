from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.admin.agent_state import AgentState
from app.agent.admin.node.runtime_context import (
    build_step_output_update,
    build_step_runtime,
    evaluate_failure_by_policy,
)
from app.agent.tools.chart_tools import CHART_TOOLS
from app.core.assistant_status import status_node
from app.core.langsmith import traceable
from app.core.llm import create_chat_model
from app.schemas.prompt import base_prompt
from app.utils.streaming_utils import invoke_with_policy, is_final_node

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
    # coordinator 模式下默认只给“任务描述 + 显式 read_from 的上游输出 + 错误信息”。
    # 这样可以显著收敛上下文，减少无关信息进入模型。
    payload: dict[str, Any] = {
        "task_description": runtime.get("task_description") or "根据已有数据生成图表",
        "upstream_outputs": runtime.get("upstream_outputs") or {},
        "read_from": runtime.get("read_from") or [],
        "errors": state.get("errors") or [],
    }
    user_input = runtime.get("user_input")
    if isinstance(user_input, str) and user_input:
        payload["user_input"] = user_input

    history_messages = runtime.get("history_messages")
    if isinstance(history_messages, list) and history_messages:
        payload["history_messages"] = history_messages

    if not runtime.get("coordinator_mode"):
        # 直连模式保持旧行为，继续透传历史上下文字段，避免行为突变。
        payload["order_context"] = state.get("order_context") or {}
        payload["excel_context"] = state.get("excel_context") or {}
        payload["results"] = state.get("results") or {}

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
    # 读取当前步骤配置（read_from、final_output、上下文开关）。
    runtime = build_step_runtime(
        state,
        "chart_agent",
        default_task_description="根据已有数据生成图表",
    )
    chart_input = _build_chart_input(state, runtime)
    failure_policy = runtime.get("failure_policy") or {}
    final_output = is_final_node(state, "chart_agent")
    try:
        llm = create_chat_model(model="qwen-plus", temperature=0.1)
        messages = [
            SystemMessage(content=_CHART_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(chart_input, ensure_ascii=False)),
        ]
        content, diagnostics = invoke_with_policy(
            llm,
            messages,
            tools=CHART_TOOLS,
            enable_stream=final_output,
            error_marker_prefix=str(
                failure_policy.get("error_marker_prefix") or "__ERROR__:"
            ),
            tool_error_counting=str(
                failure_policy.get("tool_error_counting") or "consecutive"
            ),
            max_tool_errors=int(failure_policy.get("max_tool_errors") or 2),
        )
        step_status, failed_error, content = evaluate_failure_by_policy(
            content,
            diagnostics,
            failure_policy,
        )
    except Exception:
        content = "图表服务暂时不可用，请稍后重试。"
        step_status = "failed"
        failed_error = "chart_agent 执行失败"

    results = dict(state.get("results") or {})
    results["chart"] = {
        "content": content,
        "is_end": final_output,
    }

    # chart 节点将“文本 + 结构化图表对象”同步写入 step_outputs。
    result: dict[str, Any] = {"results": results}
    result.update(
        build_step_output_update(
            runtime,
            node_name="chart_agent",
            status=step_status,
            text=content,
            output={"chart": results["chart"]},
            error=failed_error,
        )
    )
    return result
