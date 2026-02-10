from pathlib import Path
import operator
from typing import Annotated, TypedDict

import pytest
from langgraph.constants import END, START
from langgraph.graph import StateGraph

import app.agent.admin.workflow as workflow_module
from app.agent.admin.workflow import build_graph


def test_build_graph():
    png_bytes = build_graph().get_graph().draw_mermaid_png()
    assert png_bytes

    try:
        from IPython import get_ipython
        from IPython.display import Image, display

        if get_ipython() is not None:
            display(Image(png_bytes))
            return
    except Exception:
        pass

    output_path = Path("tmp/admin_workflow_graph.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(png_bytes)
    assert output_path.exists()
    print(f"graph image saved to: {output_path.resolve()}")


def _build_fake_plan(*node_names: str) -> list[dict]:
    return [
        {
            "node_name": node_name,
            "task_description": f"fake task for {node_name}",
            "last_node": ["coordinator_agent"],
        }
        for node_name in node_names
    ]


def _build_initial_state(plan: list[dict | list[dict]]) -> dict:
    return {
        "user_input": "fake user request",
        "user_intent": {"type": "fake"},
        "plan": plan,
        "routing": {},
        "order_context": {},
        "aftersale_context": {},
        "excel_context": {},
        "shared_memory": {},
        "results": {},
        "errors": [],
    }


class _VisualGraphState(TypedDict):
    # 并行节点写入时使用 reducer 聚合，避免 INVALID_CONCURRENT_GRAPH_UPDATE
    trace: Annotated[list[str], operator.add]


def _build_stage_visual_graph(stage_node_names: list[list[str]]):
    """按阶段构造可视化图，支持并行阶段（如 AB -> C）。"""
    graph = StateGraph(_VisualGraphState)

    def _noop_node(state: _VisualGraphState) -> _VisualGraphState:
        return {"trace": []}

    all_nodes = {
        "coordinator_agent",
        *(node_name for stage in stage_node_names for node_name in stage),
    }
    for node_name in all_nodes:
        graph.add_node(node_name, _noop_node)

    if not stage_node_names:
        graph.add_edge(START, END)
        return graph.compile()

    graph.add_edge(START, "coordinator_agent")
    for first_stage_node in stage_node_names[0]:
        graph.add_edge("coordinator_agent", first_stage_node)

    for stage_index in range(len(stage_node_names) - 1):
        current_stage = stage_node_names[stage_index]
        next_stage = stage_node_names[stage_index + 1]

        for next_stage_node in next_stage:
            if len(current_stage) == 1:
                graph.add_edge(current_stage[0], next_stage_node)
            else:
                graph.add_edge(current_stage, next_stage_node)

    last_stage = stage_node_names[-1]
    if len(last_stage) == 1:
        graph.add_edge(last_stage[0], END)
    else:
        graph.add_edge(last_stage, END)

    return graph.compile()


def _draw_execution_path_png(stage_nodes: list[list[str]], output_path: Path) -> None:
    """使用 LangGraph Mermaid 渲染阶段编排图，图片中可直观看到 START/END。"""
    png_bytes = _build_stage_visual_graph(stage_nodes).get_graph().draw_mermaid_png()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(png_bytes)


def _assert_execution_trace_by_stages(execution_trace: list[str], expected_stages: list[list[str]]) -> None:
    """断言阶段顺序正确；同一并行阶段内不要求节点顺序。"""
    offset = 0
    for expected_stage in expected_stages:
        stage_size = len(expected_stage)
        actual_stage = execution_trace[offset : offset + stage_size]
        assert len(actual_stage) == stage_size
        assert set(actual_stage) == set(expected_stage)
        offset += stage_size
    assert offset == len(execution_trace)


@pytest.mark.parametrize(
    ("case_name", "plan", "expected_stages"),
    [
        (
            "order_excel_chart",
            _build_fake_plan("order_agent", "excel_agent", "chart_agent"),
            [["order_agent"], ["excel_agent"], ["chart_agent"]],
        ),
        (
            "order_chart",
            _build_fake_plan("order_agent", "chart_agent"),
            [["order_agent"], ["chart_agent"]],
        ),
        (
            "order_skip_unknown_chart",
            _build_fake_plan("order_agent", "unknown_agent", "chart_agent"),
            [["order_agent"], ["chart_agent"]],
        ),
        (
            "order_and_excel_then_chart",
            [
                _build_fake_plan("order_agent", "excel_agent"),
                {
                    "node_name": "chart_agent",
                    "task_description": "final chart step",
                    "last_node": ["order_agent", "excel_agent"],
                },
            ],
            [["order_agent", "excel_agent"], ["chart_agent"]],
        ),
    ],
)
def test_workflow_dynamic_plan_sequence_and_export_path_image(
    monkeypatch: pytest.MonkeyPatch,
    case_name: str,
    plan: list[dict | list[dict]],
    expected_stages: list[list[str]],
):
    execution_trace: list[str] = []

    def fake_coordinator_agent(state: dict) -> dict:
        # 这里不改计划，直接进入 router，保证测试关注路由编排行为。
        return {}

    def _fake_node(node_name: str):
        def _runner(state: dict) -> dict:
            execution_trace.append(node_name)
            # 并行阶段避免同时写入同一个状态 key，测试只验证编排顺序。
            return {}

        return _runner

    monkeypatch.setattr(workflow_module, "coordinator", fake_coordinator_agent)
    monkeypatch.setattr(workflow_module, "order_agent", _fake_node("order_agent"))
    monkeypatch.setattr(workflow_module, "excel_agent", _fake_node("excel_agent"))
    monkeypatch.setattr(workflow_module, "chart_agent", _fake_node("chart_agent"))

    graph = workflow_module.build_graph()
    result = graph.invoke(_build_initial_state(plan))

    _assert_execution_trace_by_stages(execution_trace, expected_stages)
    assert result["routing"]["stage_index"] >= len(expected_stages)

    # 产出每个场景的编排图片: START -> coordinator_agent -> (并行/串行阶段) -> END
    output_path = Path(f"../tmp/admin_workflow_path_{case_name}.png")
    _draw_execution_path_png(expected_stages, output_path)
    assert output_path.exists()
