from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Mapping

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.utils.prompt_utils import load_prompt
from app.core.agent.agent_tool_events import tool_call_status
from app.core.agent.agent_runtime import agent_invoke
from app.core.langsmith import traceable
from app.core.llm import create_agent


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
    set(SUPPORTED_CHART_TYPES)
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
    tool_name="图表示例模板",
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


_BASE_PROMPT = load_prompt("assistant_base_prompt")
_CHART_SYSTEM_PROMPT = load_prompt("assistant_chart_system_prompt") + _BASE_PROMPT


@tool(
    description=(
        "处理图表模板相关任务：获取支持图表类型、按名称获取单个图表示例。"
        "输入为自然语言任务描述，内部会自动调用图表模板工具并返回结果。"
    )
)
@tool_call_status(
    tool_name="正在调用图表子代理",
    start_message="正在执行查询",
    error_message="调用图表子代理失败",
    timely_message="图表子代理正在持续处理中",
)
@traceable(name="Supervisor Chart Tool Agent", run_type="chain")
def chart_tool_agent(task_description: str) -> str:
    agent = create_agent(
        model="qwen-flash",
        llm_kwargs={"temperature": 0.2},
        system_prompt=SystemMessage(content=_CHART_SYSTEM_PROMPT),
        tools=[get_supported_chart_types, get_chart_sample_by_name],
    )
    input_messages = str(task_description or "").strip()
    result = agent_invoke(
        agent,
        input_messages,
    )
    return result.content


chart_agent = chart_tool_agent
