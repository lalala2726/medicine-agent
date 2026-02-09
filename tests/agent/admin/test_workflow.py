from pathlib import Path

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
            "last_node": ["supervisor_agent"],
        }
        for node_name in node_names
    ]


def _build_initial_state(plan: list[dict]) -> dict:
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


def _build_linear_visual_graph(node_names: list[str]):
    """构造仅用于可视化的线性图，导出效果与 test_build_graph 一致。"""
    graph = StateGraph(dict)

    def _noop_node(state: dict) -> dict:
        return state

    for node_name in node_names:
        graph.add_node(node_name, _noop_node)

    if not node_names:
        graph.add_edge(START, END)
    else:
        graph.add_edge(START, node_names[0])
        for idx in range(len(node_names) - 1):
            graph.add_edge(node_names[idx], node_names[idx + 1])
        graph.add_edge(node_names[-1], END)

    return graph.compile()


def _draw_execution_path_png(nodes: list[str], output_path: Path) -> None:
    """使用 LangGraph Mermaid 渲染，图片中可直观看到 START/END。"""
    png_bytes = _build_linear_visual_graph(nodes).get_graph().draw_mermaid_png()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(png_bytes)


@pytest.mark.parametrize(
    ("case_name", "plan", "expected_order"),
    [
        (
            "order_excel_chart",
            _build_fake_plan("order_agent", "excel_agent", "chart_agent"),
            ["order_agent", "excel_agent", "chart_agent"],
        ),
        (
            "order_chart",
            _build_fake_plan("order_agent", "chart_agent"),
            ["order_agent", "chart_agent"],
        ),
        (
            "order_skip_unknown_chart",
            _build_fake_plan("order_agent", "unknown_agent", "chart_agent"),
            ["order_agent", "chart_agent"],
        ),
    ],
)
def test_workflow_dynamic_plan_sequence_and_export_path_image(
    monkeypatch: pytest.MonkeyPatch,
    case_name: str,
    plan: list[dict],
    expected_order: list[str],
):
    execution_trace: list[str] = []

    def fake_supervisor_agent(state: dict) -> dict:
        # 这里不改计划，直接进入 router，保证测试关注路由编排行为。
        return {}

    def _fake_node(node_name: str):
        def _runner(state: dict) -> dict:
            execution_trace.append(node_name)
            return {"results": {node_name: {"status": "ok", "source": "fake"}}}

        return _runner

    monkeypatch.setattr(workflow_module, "supervisor_agent", fake_supervisor_agent)
    monkeypatch.setattr(workflow_module, "order_agent", _fake_node("order_agent"))
    monkeypatch.setattr(workflow_module, "excel_agent", _fake_node("excel_agent"))
    monkeypatch.setattr(workflow_module, "chart_agent", _fake_node("chart_agent"))

    graph = workflow_module.build_graph()
    result = graph.invoke(_build_initial_state(plan))

    assert execution_trace == expected_order
    assert result["routing"]["plan_index"] >= len(expected_order)

    # 产出每个场景的编排图片: START -> supervisor_agent -> ... -> END
    image_nodes = ["supervisor_agent", *execution_trace]
    output_path = Path(f"tmp/admin_workflow_path_{case_name}.png")
    _draw_execution_path_png(image_nodes, output_path)
    assert output_path.exists()
