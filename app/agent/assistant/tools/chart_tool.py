from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.core.assistant_status import tool_call_status
from app.core.langsmith import traceable
from app.core.llm import create_chat_model
from app.schemas.prompt import base_prompt
from app.utils.streaming_utils import invoke_with_trace


ChartType = Literal[
    "line",
    "area",
    "column",
    "bar",
    "pie",
    "histogram",
    "scatter",
    "wordcloud",
    "treemap",
    "dualaxes",
    "radar",
    "funnel",
    "mindmap",
    "networkgraph",
    "flowdiagram",
    "organizationchart",
    "indentedtree",
    "fishbonediagram",
]

SUPPORTED_CHART_TYPES: tuple[ChartType, ...] = (
    "line",
    "area",
    "column",
    "bar",
    "pie",
    "histogram",
    "scatter",
    "wordcloud",
    "treemap",
    "dualaxes",
    "radar",
    "funnel",
    "mindmap",
    "networkgraph",
    "flowdiagram",
    "organizationchart",
    "indentedtree",
    "fishbonediagram",
)

CHART_SAMPLES_PATH = Path(__file__).resolve().parents[4] / "resources" / "agent" / "chart_samples.json"


class ChartSampleByNameRequest(BaseModel):
    """按图表名称获取单个模板示例请求。"""

    chart_name: ChartType = Field(
        description="图表名称，必须精确匹配系统支持类型之一",
    )


@lru_cache(maxsize=1)
def _load_chart_samples() -> dict[str, dict[str, Any]]:
    """加载图表模板文件并做严格校验。"""

    if not CHART_SAMPLES_PATH.exists():
        raise FileNotFoundError(f"Chart samples file not found: {CHART_SAMPLES_PATH}")

    try:
        payload = json.loads(CHART_SAMPLES_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Chart samples file is not valid JSON: {CHART_SAMPLES_PATH}") from exc

    if not isinstance(payload, dict):
        raise ValueError(
            f"Chart samples file must be a JSON object mapping chart type to sample: {CHART_SAMPLES_PATH}"
        )

    # 忽略以 _ 开头的元数据字段（如 _meta）
    meta_keys = {k for k in payload.keys() if k.startswith("_")}
    payload_keys = set(payload.keys()) - meta_keys
    expected_keys = set(SUPPORTED_CHART_TYPES)
    missing = [chart for chart in SUPPORTED_CHART_TYPES if chart not in payload_keys]
    if missing:
        raise ValueError(f"Chart samples keys mismatch: missing={missing}")

    normalized: dict[str, dict[str, Any]] = {}
    for chart in SUPPORTED_CHART_TYPES:
        sample = payload.get(chart)
        if not isinstance(sample, dict):
            raise ValueError(f"Chart sample '{chart}' must be a JSON object")
        # 过滤掉以 _ 开头的元数据字段（如 _description, _fields, _output_format）
        filtered_sample = {k: v for k, v in sample.items() if not k.startswith("_")}
        normalized[chart] = filtered_sample

    return normalized


@tool(description="获取当前系统支持的图表类型列表（全量18种）。")
@tool_call_status(
    tool_name="获取支持图表类型",
    start_message="正在获取系统支持的图表类型",
    error_message="获取支持图表类型失败",
    timely_message="图表类型正在持续处理中",
)
def get_supported_chart_types() -> dict:
    """返回系统支持的图表类型。"""

    return {
        "chart_types": list(SUPPORTED_CHART_TYPES),
        "count": len(SUPPORTED_CHART_TYPES),
    }


@tool(
    args_schema=ChartSampleByNameRequest,
    description=(
        "按图表名称获取单个图表模板示例及字段说明。"
        "必须精确传入受支持图表名，返回示例、字段定义和输出格式。"
    ),
)
@tool_call_status(
    tool_name="get_chart_sample_by_name",
    start_message="正在获取图表配置模板",
    error_message="获取图表模板失败",
    timely_message="图表模板正在持续处理中",
)
def get_chart_sample_by_name(chart_name: ChartType) -> dict:
    """按名称获取单个图表示例模板，包含字段说明和输出格式。"""

    normalized_name = str(chart_name or "").strip()
    if normalized_name not in SUPPORTED_CHART_TYPES:
        raise ValueError(
            f"不支持的图表类型: {normalized_name}。"
            f"仅支持: {', '.join(SUPPORTED_CHART_TYPES)}"
        )

    # 直接从文件读取完整的模板信息（包括元数据）
    if not CHART_SAMPLES_PATH.exists():
        raise FileNotFoundError(f"Chart samples file not found: {CHART_SAMPLES_PATH}")

    try:
        payload = json.loads(CHART_SAMPLES_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Chart samples file is not valid JSON") from exc

    sample = payload.get(normalized_name)
    if not isinstance(sample, dict):
        raise ValueError(f"图表类型 {normalized_name} 未配置合法示例")

    # 提取元数据字段
    description = sample.get("_description", "")
    fields = sample.get("_fields", {})
    output_format = sample.get("_output_format", "")

    # 过滤掉元数据字段，只保留实际配置
    config_sample = {k: v for k, v in sample.items() if not k.startswith("_")}

    return {
        "chart_type": normalized_name,
        "description": description,
        "fields": fields,
        "output_format": output_format,
        "sample": copy.deepcopy(config_sample),
    }


_CHART_SYSTEM_PROMPT = (
    """
        你是图表模板域子工具（chart_tool_agent），只负责图表类型与图表模板问题。

        ## 职责范围
        1. 查询系统支持的图表类型（共18种）。
        2. 根据图表名称获取单个图表示例模板，包含字段说明。

        ## 支持的图表类型（18种）
        | 图表类型 | 代码块标识 | 用途说明 |
        |---------|-----------|---------|
        | line | ```line | 折线图 - 展示趋势变化 |
        | area | ```area | 面积图 - 趋势+总量 |
        | column | ```column | 柱状图 - 分类比较（垂直） |
        | bar | ```bar | 条形图 - 分类比较（水平） |
        | pie | ```pie | 饼图 - 占比展示 |
        | histogram | ```histogram | 直方图 - 数据分布 |
        | scatter | ```scatter | 散点图 - 变量关系 |
        | wordcloud | ```wordcloud | 词云图 - 词频展示 |
        | treemap | ```treemap | 矩阵树图 - 层级占比 |
        | dualaxes | ```dualaxes | 双轴图 - 多量纲组合 |
        | radar | ```radar | 雷达图 - 多维评价 |
        | funnel | ```funnel | 漏斗图 - 转化分析 |
        | mindmap | ```mindmap | 思维导图 - 思维发散 |
        | networkgraph | ```networkgraph | 关系图 - 节点关系 |
        | flowdiagram | ```flowdiagram | 流程图 - 步骤流程 |
        | organizationchart | ```organizationchart | 组织架构图 - 层级结构 |
        | indentedtree | ```indentedtree | 缩进树图 - 目录结构 |
        | fishbonediagram | ```fishbonediagram | 鱼骨图 - 因果分析 |

        ## 图表输出规则（严格遵守）
        1. 必须先调用 get_supported_chart_types 确认支持的图表类型。
        2. 然后调用 get_chart_sample_by_name 获取目标图表的模板和字段说明。
        3. 输出格式必须使用 Markdown 代码块：
           - 代码块开头的语言标识必须精确匹配图表类型（如 ```line、```pie）
           - 代码块内容必须是合法的 JSON
        4. 字段结构必须严格遵循模板返回的 _fields 说明：
           - 必填字段不能省略
           - 数据类型必须正确（文本用字符串，数值用数字）
           - 字段名大小写必须精确匹配
        5. 禁止添加模板中不存在的字段。

        ## 数据填写规范
        - time/category/name 等文本字段：使用有意义的业务名称
        - value/数值字段：使用真实数值，禁止包含单位符号或数学运算
        - group 字段：用于多系列对比时填写分组名称
        - 嵌套结构（children）：保持正确的缩进和层级

        ## 强约束
        1. 严禁编造不支持的图表类型。
        2. 严禁修改模板字段结构。
        3. 输出简洁，先说明选用图表类型及原因，再给出代码块。
        4. 如用户需求不明确，主动询问需要展示的数据维度和图表类型。
    """
    + base_prompt
)


@tool(
    description=(
        "处理图表模板相关任务：获取支持图表类型、按名称获取单个图表示例。"
        "输入为自然语言任务描述，内部会自动调用图表模板工具并返回结果。"
    )
)
@traceable(name="Supervisor Chart Tool Agent", run_type="chain")
def chart_tool_agent(task_description: str) -> str:
    llm = create_chat_model(
        model="qwen-flash",
        temperature=0.2,
    )
    messages = [
        SystemMessage(content=_CHART_SYSTEM_PROMPT),
        HumanMessage(content=str(task_description or "").strip()),
    ]
    trace = invoke_with_trace(
        llm,
        messages,
        tools=[get_supported_chart_types, get_chart_sample_by_name],
    )
    text = str(trace.get("text") or "").strip()
    if not text:
        return "未获取到图表模板数据，请补充图表名称或场景后重试。"
    return text


chart_agent = chart_tool_agent
