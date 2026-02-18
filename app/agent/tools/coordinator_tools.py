from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class AgentDetailRequest(BaseModel):
    """
    `get_agent_detail` 工具参数定义。
    """

    agent_names: list[str] = Field(
        ...,
        description=(
            "要查询的节点名称列表。可选值：order_agent、product_agent、chart_agent、"
            "summary_agent、excel_agent；也可传 ['all'] 查询全部。"
        ),
        min_length=1,
    )
    include_tool_parameters: bool = Field(
        default=True,
        description="是否返回节点可用工具及工具参数说明。",
    )
    include_coordination_guide: bool = Field(
        default=True,
        description="是否返回节点编排与协同建议（上游依赖、推荐读源、输出用途）。",
    )
    include_plan_examples: bool = Field(
        default=False,
        description="是否返回该节点在 DAG 计划中的示例片段。",
    )


_COMMON_STEP_PARAMETERS = {
    "step_id": "步骤唯一标识，字符串，必须全局唯一。",
    "node_name": "节点名称，必须是可执行节点之一。",
    "task_description": "该步骤执行任务描述，建议写可执行动词和目标。",
    "required_depends_on": "必选依赖步骤 step_id 数组；仅当依赖全部 completed 才会执行。",
    "optional_depends_on": "可选依赖步骤 step_id 数组；允许失败/跳过，但需先进入终态后当前步骤才能执行。",
    "read_from": "读取上游输出的 step_id 数组；必须是当前步骤可达上游。",
    "include_user_input": "是否注入原始用户输入（默认 false）。",
    "include_chat_history": "是否注入历史 user/assistant 消息（默认 false）。",
    "final_output": "是否作为最终输出步骤（全计划必须且仅能一个 true）。",
    "failure_policy": (
        "步骤级失败策略（可选）。推荐："
        "{mode: hybrid, error_marker_prefix: '__ERROR__:', "
        "tool_error_counting: consecutive, max_tool_errors: 2, strict_data_quality: true}。"
    ),
}

_DEFAULT_FAILURE_POLICY = {
    "mode": "hybrid",
    "error_marker_prefix": "__ERROR__:",
    "tool_error_counting": "consecutive",
    "max_tool_errors": 2,
    "strict_data_quality": True,
}

_AGENT_DETAIL_CATALOG: dict[str, dict[str, Any]] = {
    "order_agent": {
        "summary": "处理订单查询、订单校验、订单状态判断等订单域任务。",
        "typical_tasks": [
            "按订单号/状态/收货人过滤查询订单列表",
            "核验订单是否存在、状态是否满足业务条件",
            "按订单 ID 拉取订单明细",
        ],
        "available_tools": [
            {
                "name": "get_order_list",
                "purpose": "分页查询订单列表，支持多条件筛选。",
                "parameters": {
                    "page_num": "页码（默认 1）",
                    "page_size": "每页数量（默认 10）",
                    "order_no": "订单编号",
                    "pay_type": "支付方式编码",
                    "order_status": "订单状态编码",
                    "delivery_type": "配送方式编码",
                    "receiver_name": "收货人姓名",
                    "receiver_phone": "收货人手机号",
                },
            },
            {
                "name": "get_orders_detail",
                "purpose": "根据订单ID获取订单详情，支持批量查询。",
                "parameters": {
                    "order_id": "订单ID（必填），支持单个或多个",
                },
            },
        ],
        "coordination_guide": {
            "recommended_upstream": [],
            "recommended_downstream": ["product_agent", "chart_agent", "summary_agent"],
            "recommended_read_from": [],
            "notes": [
                "通常作为数据起点步骤（required_depends_on=[]）。",
                "若下游需要基于订单商品ID查商品，建议 product_agent read_from 当前步骤。",
            ],
        },
    },
    "product_agent": {
        "summary": "处理商品列表查询、商品详情查询、药品详情查询和商品信息核验。",
        "typical_tasks": [
            "按商品名、分类、状态、价格区间查询商品列表",
            "根据商品 ID 拉取商品详情",
            "根据药品商品 ID 拉取药品详细信息（说明书、适应症、用法用量等）",
            "核验商品状态、库存或价格信息",
        ],
        "available_tools": [
            {
                "name": "get_product_list",
                "purpose": "分页查询商品列表，支持组合筛选。",
                "parameters": {
                    "page_num": "页码（默认 1）",
                    "page_size": "每页数量（默认 10）",
                    "id": "商品 ID",
                    "name": "商品名称关键词",
                    "category_id": "商品分类 ID",
                    "status": "上/下架状态（1/0）",
                    "min_price": "最低价格",
                    "max_price": "最高价格",
                },
            },
            {
                "name": "get_product_info",
                "purpose": "根据商品ID查询商品详情，支持批量查询。",
                "parameters": {
                    "product_id": "商品ID（必填），支持单个或多个",
                },
            },
            {
                "name": "get_drug_detail",
                "purpose": "根据药品商品ID查询药品详细信息，包括说明书、适应症、用法用量等，支持批量查询。",
                "parameters": {
                    "product_id": "药品商品ID列表（必填），传入字符串列表形式，如 ['1001'] 或 ['1001', '1002']",
                },
            },
        ],
        "coordination_guide": {
            "recommended_upstream": ["order_agent"],
            "recommended_downstream": ["chart_agent", "summary_agent"],
            "recommended_read_from": ["order_agent"],
            "notes": [
                "常见链路：order_agent -> product_agent（按订单中的商品ID补充商品信息）。",
            ],
        },
    },
    "chart_agent": {
        "summary": "基于已有结构化数据生成可渲染图表配置。",
        "typical_tasks": [
            "将订单/商品聚合数据转为折线、柱状、饼图等图表配置",
            "按任务要求生成指定类型图表模板并填充数据",
        ],
        "available_tools": [
            {
                "name": "get_chart_sample_by_name",
                "purpose": "根据图表类型获取标准模板，再替换数据值。",
                "parameters": {
                    "explanation": "调用原因说明（必填）",
                    "name_or_type": "图表名称或类型（如 line、pie、column）",
                },
            }
        ],
        "coordination_guide": {
            "recommended_upstream": ["order_agent", "product_agent"],
            "recommended_downstream": ["summary_agent"],
            "recommended_read_from": ["order_agent", "product_agent"],
            "notes": [
                "不应作为首步骤编造数据，建议读取上游真实数据后再作图。",
            ],
        },
    },
    "summary_agent": {
        "summary": "汇总多个节点结果并输出最终结论。",
        "typical_tasks": [
            "整合订单、商品、图表节点结果形成最终答复",
            "输出结论并标注关键依据与待确认项",
        ],
        "available_tools": [],
        "coordination_guide": {
            "recommended_upstream": ["order_agent", "product_agent", "chart_agent", "excel_agent"],
            "recommended_downstream": [],
            "recommended_read_from": ["order_agent", "product_agent", "chart_agent", "excel_agent"],
            "notes": [
                "通常设置 final_output=true。",
                "建议作为终点步骤，不再被其他步骤 required/optional 依赖。",
            ],
        },
    },
    "excel_agent": {
        "summary": "表格能力节点（当前仅保留编排占位，业务能力待实现）。",
        "typical_tasks": [
            "未来用于表格导出和结构化表格处理",
        ],
        "available_tools": [],
        "coordination_guide": {
            "recommended_upstream": ["order_agent", "product_agent"],
            "recommended_downstream": ["summary_agent"],
            "recommended_read_from": ["order_agent", "product_agent"],
            "notes": [
                "当前节点执行会返回未实现失败，不建议在关键主链中依赖其成功。",
            ],
        },
    },
}

_PLAN_EXAMPLES_BY_AGENT: dict[str, dict[str, Any]] = {
    "order_agent": {
        "step_id": "s_order",
        "node_name": "order_agent",
        "task_description": "查询订单并提取关键字段",
        "required_depends_on": [],
        "optional_depends_on": [],
        "read_from": [],
        "include_user_input": False,
        "include_chat_history": False,
        "final_output": False,
        "failure_policy": _DEFAULT_FAILURE_POLICY,
    },
    "product_agent": {
        "step_id": "s_product",
        "node_name": "product_agent",
        "task_description": "基于订单中的商品ID查询商品详情",
        "required_depends_on": ["s_order"],
        "optional_depends_on": [],
        "read_from": ["s_order"],
        "include_user_input": False,
        "include_chat_history": False,
        "final_output": False,
        "failure_policy": _DEFAULT_FAILURE_POLICY,
    },
    "chart_agent": {
        "step_id": "s_chart",
        "node_name": "chart_agent",
        "task_description": "根据订单与商品数据生成趋势图",
        "required_depends_on": ["s_product"],
        "optional_depends_on": ["s_order"],
        "read_from": ["s_order", "s_product"],
        "include_user_input": False,
        "include_chat_history": False,
        "final_output": False,
        "failure_policy": _DEFAULT_FAILURE_POLICY,
    },
    "summary_agent": {
        "step_id": "s_summary",
        "node_name": "summary_agent",
        "task_description": "汇总上游节点结果并输出最终结论",
        "required_depends_on": ["s_product", "s_chart"],
        "optional_depends_on": ["s_order"],
        "read_from": ["s_order", "s_product", "s_chart"],
        "include_user_input": False,
        "include_chat_history": False,
        "final_output": True,
        "failure_policy": _DEFAULT_FAILURE_POLICY,
    },
    "excel_agent": {
        "step_id": "s_excel",
        "node_name": "excel_agent",
        "task_description": "导出上游数据为表格（占位示例）",
        "required_depends_on": ["s_order"],
        "optional_depends_on": [],
        "read_from": ["s_order"],
        "include_user_input": False,
        "include_chat_history": False,
        "final_output": False,
        "failure_policy": _DEFAULT_FAILURE_POLICY,
    },
}


def _normalize_agent_names(agent_names: list[str]) -> list[str]:
    """
    规范化节点名称列表，去重并保留顺序。

    Args:
        agent_names: 原始节点名列表。

    Returns:
        list[str]: 归一化后的节点名列表（小写、去空、去重且保序）。
    """
    normalized: list[str] = []
    seen: set[str] = set()
    for name in agent_names:
        key = str(name).strip().lower()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    return normalized


@tool(
    args_schema=AgentDetailRequest,
    description=(
        "获取一个或多个节点的详细能力说明、可用工具与参数、输入输出约束、"
        "编排协同建议与计划片段示例。"
    ),
)
def get_agent_detail(
        agent_names: list[str],
        include_tool_parameters: bool = True,
        include_coordination_guide: bool = True,
        include_plan_examples: bool = False,
) -> dict[str, Any]:
    """
    返回 coordinator 规划所需的节点能力目录。

    Args:
        agent_names: 目标节点名列表，支持传 `["all"]` 查询全部。
        include_tool_parameters: 是否包含工具参数说明。
        include_coordination_guide: 是否包含编排协同建议。
        include_plan_examples: 是否包含 DAG 片段示例。

    Returns:
        dict[str, Any]:
            - `ok`: 是否成功
            - `agent_details`: 节点能力详情
            - 以及 requested/resolved/unsupported 等辅助字段
    """
    requested = _normalize_agent_names(agent_names)
    if not requested:
        return {
            "ok": False,
            "message": "agent_names 不能为空，请至少传入一个节点名称。",
            "supported_agents": sorted(_AGENT_DETAIL_CATALOG.keys()),
        }

    if "all" in requested:
        resolved_names = list(_AGENT_DETAIL_CATALOG.keys())
    else:
        resolved_names = [name for name in requested if name in _AGENT_DETAIL_CATALOG]

    unsupported = [name for name in requested if name not in _AGENT_DETAIL_CATALOG and name != "all"]
    if not resolved_names:
        return {
            "ok": False,
            "message": "未匹配到有效节点名称，请检查 agent_names。",
            "requested_agents": requested,
            "unsupported_agents": unsupported,
            "supported_agents": sorted(_AGENT_DETAIL_CATALOG.keys()),
        }

    details: dict[str, Any] = {}
    for name in resolved_names:
        base_detail = dict(_AGENT_DETAIL_CATALOG[name])
        if not include_tool_parameters:
            base_detail.pop("available_tools", None)
        if not include_coordination_guide:
            base_detail.pop("coordination_guide", None)
        if include_plan_examples:
            base_detail["plan_example"] = _PLAN_EXAMPLES_BY_AGENT.get(name, {})
        details[name] = base_detail

    return {
        "ok": True,
        "requested_agents": requested,
        "resolved_agents": resolved_names,
        "unsupported_agents": unsupported,
        "supported_agents": sorted(_AGENT_DETAIL_CATALOG.keys()),
        "tool_parameters_supported": {
            "agent_names": "list[str]，必填，节点名称数组；可传 ['all']。",
            "include_tool_parameters": "bool，默认 true，返回节点可用工具及参数说明。",
            "include_coordination_guide": "bool，默认 true，返回节点编排协同建议。",
            "include_plan_examples": "bool，默认 false，返回节点计划片段示例。",
        },
        "common_plan_step_parameters": _COMMON_STEP_PARAMETERS,
        "agent_details": details,
        "usage_suggestion": (
            "建议按需分批查询节点详情，避免一次拉取全部内容导致规划上下文过长。"
        ),
    }
