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

    payload_keys = set(payload.keys())
    expected_keys = set(SUPPORTED_CHART_TYPES)
    missing = [chart for chart in SUPPORTED_CHART_TYPES if chart not in payload_keys]
    extra = sorted(payload_keys - expected_keys)
    if missing or extra:
        raise ValueError(f"Chart samples keys mismatch: missing={missing}, extra={extra}")

    normalized: dict[str, dict[str, Any]] = {}
    for chart in SUPPORTED_CHART_TYPES:
        sample = payload.get(chart)
        if not isinstance(sample, dict):
            raise ValueError(f"Chart sample '{chart}' must be a JSON object")
        normalized[chart] = sample

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
        "按图表名称获取单个图表模板示例。"
        "必须精确传入受支持图表名，仅返回一个示例，避免上下文膨胀。"
    ),
)
@tool_call_status(
    tool_name="get_chart_sample_by_name",
    start_message="正在获取图表配置模板",
    error_message="获取图表模板失败",
    timely_message="图表模板正在持续处理中",
)
def get_chart_sample_by_name(chart_name: ChartType) -> dict:
    """按名称获取单个图表示例模板。"""

    normalized_name = str(chart_name or "").strip()
    if normalized_name not in SUPPORTED_CHART_TYPES:
        raise ValueError(
            f"不支持的图表类型: {normalized_name}。"
            f"仅支持: {', '.join(SUPPORTED_CHART_TYPES)}"
        )

    samples = _load_chart_samples()
    sample = samples.get(normalized_name)
    if not isinstance(sample, dict):
        raise ValueError(f"图表类型 {normalized_name} 未配置合法示例")

    return {
        "chart_type": normalized_name,
        "sample": copy.deepcopy(sample),
    }


_CHART_SYSTEM_PROMPT = (
    """
        你是图表模板域子工具（chart_tool_agent），只负责图表类型与图表模板问题。

        你只能处理：
        1. 查询系统支持的图表类型。
        2. 根据图表名称获取单个图表示例模板。

        图表输出硬性规则：
        1. 当用户要求输出图表时，必须先调用 get_supported_chart_types。
        2. 然后根据目标图表名称，只能调用一次 get_chart_sample_by_name 获取单个模板。
        3. 输出图表必须使用 Markdown 代码块，language 必须等于 chart_type。
        4. 代码块内容必须是可 JSON.parse 的合法 JSON。
        5. 字段结构与层级必须严格遵循 sample，不允许修改模板结构。

        强约束：
        1. 严禁凭空编造图表类型或模板字段。
        2. 输出简洁，先给结论，再给代码块。
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
