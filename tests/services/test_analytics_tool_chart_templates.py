from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agent.assistant.tools import chart_tool

EXPECTED_CHART_TYPES = [
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


def _build_valid_samples() -> dict[str, dict]:
    return {name: {"name": name, "data": []} for name in EXPECTED_CHART_TYPES}


def test_get_supported_chart_types_returns_full_list() -> None:
    result = chart_tool.get_supported_chart_types.invoke({})

    assert result["chart_types"] == EXPECTED_CHART_TYPES
    assert result["count"] == len(EXPECTED_CHART_TYPES)


def test_get_chart_sample_by_name_returns_single_sample_and_deep_copy() -> None:
    chart_tool._load_chart_samples.cache_clear()

    first = chart_tool.get_chart_sample_by_name.invoke({"chart_name": "line"})
    assert first["chart_type"] == "line"
    assert isinstance(first["sample"], dict)

    first["sample"]["data"][0]["value"] = -1

    second = chart_tool.get_chart_sample_by_name.invoke({"chart_name": "line"})
    assert second["chart_type"] == "line"
    assert second["sample"]["data"][0]["value"] != -1


def test_get_chart_sample_by_name_rejects_invalid_name() -> None:
    with pytest.raises(Exception):
        chart_tool.get_chart_sample_by_name.invoke({"chart_name": "Line"})

    with pytest.raises(ValueError, match="不支持的图表类型"):
        chart_tool.get_chart_sample_by_name.func("line_chart")


def test_load_chart_samples_raises_when_file_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chart_tool._load_chart_samples.cache_clear()
    monkeypatch.setattr(chart_tool, "CHART_SAMPLES_PATH", tmp_path / "missing.json")

    with pytest.raises(FileNotFoundError):
        chart_tool._load_chart_samples()

    chart_tool._load_chart_samples.cache_clear()


def test_load_chart_samples_raises_on_invalid_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    invalid_file = tmp_path / "chart_samples.json"
    invalid_file.write_text("{not-json}", encoding="utf-8")

    chart_tool._load_chart_samples.cache_clear()
    monkeypatch.setattr(chart_tool, "CHART_SAMPLES_PATH", invalid_file)

    with pytest.raises(ValueError, match="not valid JSON"):
        chart_tool._load_chart_samples()

    chart_tool._load_chart_samples.cache_clear()


def test_load_chart_samples_raises_on_key_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mismatch_file = tmp_path / "chart_samples.json"
    payload = _build_valid_samples()
    payload.pop("line")
    payload["unknown"] = {"name": "unknown", "data": []}
    mismatch_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    chart_tool._load_chart_samples.cache_clear()
    monkeypatch.setattr(chart_tool, "CHART_SAMPLES_PATH", mismatch_file)

    with pytest.raises(ValueError, match="keys mismatch"):
        chart_tool._load_chart_samples()

    chart_tool._load_chart_samples.cache_clear()
