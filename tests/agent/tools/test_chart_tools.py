import pytest

import app.agent.tools.chart_tools as chart_tools_module


def test_chart_tools_only_exposes_sample_tool():
    assert len(chart_tools_module.CHART_TOOLS) == 1
    assert chart_tools_module.CHART_TOOLS[0].name == "get_chart_sample_by_name"


def test_get_chart_sample_by_name_supports_type_lookup():
    payload = chart_tools_module.get_chart_sample_by_name.invoke(
        {"explanation": "趋势图", "name_or_type": "line"}
    )

    assert isinstance(payload, dict)
    assert "data" in payload
    assert "axisXTitle" in payload
    assert "axisYTitle" in payload


def test_get_chart_sample_by_name_supports_name_lookup():
    payload = chart_tools_module.get_chart_sample_by_name.invoke(
        {"explanation": "按中文名获取模板", "name_or_type": "饼图"}
    )

    assert isinstance(payload, dict)
    assert "data" in payload
    assert "height" in payload


def test_get_chart_sample_by_name_raises_when_missing():
    with pytest.raises(ValueError, match="未找到对应的图表类型"):
        chart_tools_module.get_chart_sample_by_name.invoke(
            {"explanation": "未知类型", "name_or_type": "unknown_chart"}
        )
