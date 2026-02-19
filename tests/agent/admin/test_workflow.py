from pathlib import Path

import pytest

import app.agent.admin.node.chat_node as chat_module
import app.agent.admin.workflow as workflow_module
from app.agent.admin.workflow import build_graph


def _safe_draw_mermaid_png(compiled_graph) -> bytes:
    try:
        return compiled_graph.get_graph().draw_mermaid_png()
    except Exception as exc:
        return f"mermaid render skipped: {exc}".encode("utf-8")


def _build_initial_state(
        plan: list[dict],
        routing: dict | None = None,
        step_outputs: dict | None = None,
) -> dict:
    return {
        "user_input": "fake user request",
        "user_intent": {"type": "fake"},
        "plan": plan,
        "routing": routing or {},
        "history_messages": [],
        "step_outputs": step_outputs or {},
        "execution_traces": [],
        "shared_memory": {},
        "results": {},
        "errors": [],
    }


def test_build_graph():
    png_bytes = _safe_draw_mermaid_png(build_graph())
    assert png_bytes

    output_path = Path("tmp/admin_workflow_graph.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(png_bytes)
    assert output_path.exists()


def test_chat_node_handles_non_business_chat(monkeypatch: pytest.MonkeyPatch):
    class _DummyResponse:
        def __init__(self, content: str):
            self.content = content

    class _DummyModel:
        def invoke(self, _messages):
            return _DummyResponse("你好，我可以陪你聊聊日常问题。")

    monkeypatch.setattr(chat_module, "create_chat_model", lambda *args, **kwargs: _DummyModel())

    state = _build_initial_state(plan=[])
    state["user_input"] = "今天天气有点冷，给点保暖建议"
    result = chat_module.chat_agent(state)
    assert result["results"]["chat"]["mode"] == "chat"
    assert result["results"]["chat"]["content"] == "你好，我可以陪你聊聊日常问题。"
    assert result["execution_traces"][0]["node_name"] == "chat_agent"


def test_chat_node_streams_when_model_supports_stream(monkeypatch: pytest.MonkeyPatch):
    class _DummyChunk:
        def __init__(self, content: str):
            self.content = content

    class _DummyModel:
        def __init__(self):
            self.invoke_called = False

        def stream(self, _messages):
            yield _DummyChunk("你")
            yield _DummyChunk("好")

        def invoke(self, _messages):
            self.invoke_called = True
            return _DummyChunk("你好")

    dummy_model = _DummyModel()
    monkeypatch.setattr(chat_module, "create_chat_model", lambda *args, **kwargs: dummy_model)

    state = _build_initial_state(plan=[])
    state["user_input"] = "你好"
    result = chat_module.chat_agent(state)
    assert dummy_model.invoke_called is False
    assert result["results"]["chat"]["content"] == "你好"
    assert result["execution_traces"][0]["model_name"] == "qwen-flash"


def test_gateway_and_planner_emit_execution_trace(monkeypatch: pytest.MonkeyPatch):
    """测试目标：网关与规划节点写入 trace；成功标准：节点名与空输入/空工具字段符合约定。"""

    class _DummyResponse:
        def __init__(self, content: str):
            self.content = content

    class _DummyModel:
        def invoke(self, _messages):
            return _DummyResponse('{"route_target":"chat_agent","difficulty":"simple"}')

    monkeypatch.setattr(workflow_module, "create_chat_model", lambda **_kwargs: _DummyModel())
    monkeypatch.setattr(workflow_module, "compute_planner_update", lambda _state: {"routing": {"next_nodes": []}})

    gateway_update = workflow_module.gateway_router(_build_initial_state(plan=[]))
    planner_update = workflow_module.planner(_build_initial_state(plan=[]))

    assert gateway_update["execution_traces"][0]["node_name"] == "gateway_router"
    assert gateway_update["execution_traces"][0]["input_messages"] == []
    assert gateway_update["execution_traces"][0]["tool_calls"] == []
    assert planner_update["execution_traces"][0]["node_name"] == "planner"
    assert planner_update["execution_traces"][0]["input_messages"] == []
    assert planner_update["execution_traces"][0]["tool_calls"] == []


def _make_step(
        step_id: str,
        node_name: str,
        *,
        depends_on: list[str] | None = None,
        optional_depends_on: list[str] | None = None,
        read_from: list[str] | None = None,
        final_output: bool = False,
) -> dict:
    return {
        "step_id": step_id,
        "node_name": node_name,
        "task_description": f"task for {step_id}",
        "required_depends_on": depends_on or [],
        "optional_depends_on": optional_depends_on or [],
        "read_from": read_from or [],
        "include_user_input": False,
        "include_chat_history": False,
        "final_output": final_output,
    }


@pytest.mark.parametrize(
    ("case_name", "plan", "expected_trace"),
    [
        (
            "a_to_b_to_c",
            [
                _make_step("s1", "order_agent"),
                _make_step("s2", "product_agent", depends_on=["s1"], read_from=["s1"]),
                _make_step(
                    "s3",
                    "summary_agent",
                    depends_on=["s2"],
                    read_from=["s2"],
                    final_output=True,
                ),
            ],
            ["order_agent", "product_agent", "summary_agent"],
        ),
        (
            "a_parallel_b_then_c",
            [
                _make_step("s1", "order_agent"),
                _make_step("s2", "product_agent"),
                _make_step(
                    "s3",
                    "summary_agent",
                    depends_on=["s1", "s2"],
                    read_from=["s1", "s2"],
                    final_output=True,
                ),
            ],
            ["order_agent", "product_agent", "summary_agent"],
        ),
        (
            "same_node_twice_sequential",
            [
                _make_step("s1", "product_agent"),
                _make_step("s2", "product_agent", depends_on=["s1"], read_from=["s1"]),
                _make_step(
                    "s3",
                    "summary_agent",
                    depends_on=["s2"],
                    read_from=["s2"],
                    final_output=True,
                ),
            ],
            ["product_agent", "product_agent", "summary_agent"],
        ),
    ],
)
def test_workflow_dynamic_dag_sequence(
        monkeypatch: pytest.MonkeyPatch,
        case_name: str,
        plan: list[dict],
        expected_trace: list[str],
):
    execution_trace: list[str] = []

    def fake_gateway_router(_state: dict) -> dict:
        return {
            "routing": {
                "route_target": "coordinator_agent",
                "difficulty": "medium",
                "next_nodes": [],
            }
        }

    def fake_coordinator(_state: dict) -> dict:
        return {}

    def _fake_node(node_name: str):
        def _runner(state: dict) -> dict:
            execution_trace.append(node_name)
            step = ((state.get("routing") or {}).get("current_step_map") or {}).get(node_name) or {}
            step_id = step.get("step_id")
            if not step_id:
                return {}
            return {
                "step_outputs": {
                    step_id: {
                        "step_id": step_id,
                        "node_name": node_name,
                        "status": "completed",
                        "text": f"{node_name} done",
                        "output": {"ok": True},
                    }
                }
            }

        return _runner

    monkeypatch.setattr(workflow_module, "gateway_router", fake_gateway_router)
    monkeypatch.setattr(workflow_module, "coordinator", fake_coordinator)
    monkeypatch.setattr(workflow_module, "order_agent", _fake_node("order_agent"))
    monkeypatch.setattr(workflow_module, "product_agent", _fake_node("product_agent"))
    monkeypatch.setattr(workflow_module, "summary_agent", _fake_node("summary_agent"))
    monkeypatch.setattr(workflow_module, "chart_agent", _fake_node("chart_agent"))
    monkeypatch.setattr(workflow_module, "excel_agent", _fake_node("excel_agent"))

    graph = workflow_module.build_graph()
    result = graph.invoke(_build_initial_state(plan=plan))

    assert execution_trace == expected_trace
    assert result["routing"]["next_nodes"] == []
    assert all(step_id in result["step_outputs"] for step_id in [step["step_id"] for step in plan])

    output_path = Path(f"tmp/admin_workflow_path_{case_name}.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(",".join(execution_trace), encoding="utf-8")
    assert output_path.exists()


def test_workflow_blocks_downstream_on_failure(monkeypatch: pytest.MonkeyPatch):
    plan = [
        _make_step("s1", "order_agent"),
        _make_step("s2", "summary_agent", depends_on=["s1"], read_from=["s1"], final_output=True),
    ]

    def fake_gateway_router(_state: dict) -> dict:
        return {"routing": {"route_target": "coordinator_agent", "difficulty": "medium"}}

    def fake_coordinator(_state: dict) -> dict:
        return {}

    def fail_order(state: dict) -> dict:
        step = ((state.get("routing") or {}).get("current_step_map") or {}).get("order_agent") or {}
        step_id = step.get("step_id")
        return {
            "step_outputs": {
                step_id: {
                    "step_id": step_id,
                    "node_name": "order_agent",
                    "status": "failed",
                    "text": "fail",
                    "output": {},
                    "error": "boom",
                }
            }
        }

    monkeypatch.setattr(workflow_module, "gateway_router", fake_gateway_router)
    monkeypatch.setattr(workflow_module, "coordinator", fake_coordinator)
    monkeypatch.setattr(workflow_module, "order_agent", fail_order)
    monkeypatch.setattr(workflow_module, "summary_agent", lambda _state: {})
    monkeypatch.setattr(workflow_module, "chat_agent", lambda _state: {"results": {"chat": {"content": "fallback"}}})

    result = workflow_module.build_graph().invoke(_build_initial_state(plan=plan))
    assert result["step_outputs"]["s2"]["status"] == "skipped"
    assert "阻断" in result["step_outputs"]["s2"]["error"]
    assert result["routing"]["next_nodes"] == ["chat_agent"] or "fallback_context" in result["routing"]


def test_workflow_merges_results_on_parallel_nodes(monkeypatch: pytest.MonkeyPatch):
    plan = [
        _make_step("s1", "order_agent"),
        _make_step("s2", "product_agent"),
        _make_step(
            "s3",
            "summary_agent",
            depends_on=["s1", "s2"],
            read_from=["s1", "s2"],
            final_output=True,
        ),
    ]

    def fake_gateway_router(_state: dict) -> dict:
        return {"routing": {"route_target": "coordinator_agent", "difficulty": "medium"}}

    def fake_coordinator(_state: dict) -> dict:
        return {}

    def _parallel_node(node_name: str, result_key: str):
        def _runner(state: dict) -> dict:
            step = ((state.get("routing") or {}).get("current_step_map") or {}).get(node_name) or {}
            step_id = step.get("step_id")
            return {
                "results": {result_key: {"content": f"{node_name} done"}},
                "step_outputs": {
                    step_id: {
                        "step_id": step_id,
                        "node_name": node_name,
                        "status": "completed",
                        "text": f"{node_name} done",
                        "output": {"ok": True},
                    }
                },
            }

        return _runner

    def summary_runner(state: dict) -> dict:
        step = ((state.get("routing") or {}).get("current_step_map") or {}).get("summary_agent") or {}
        step_id = step.get("step_id")
        return {
            "results": {"summary": {"content": "summary done"}},
            "step_outputs": {
                step_id: {
                    "step_id": step_id,
                    "node_name": "summary_agent",
                    "status": "completed",
                    "text": "summary done",
                    "output": {"ok": True},
                }
            },
        }

    monkeypatch.setattr(workflow_module, "gateway_router", fake_gateway_router)
    monkeypatch.setattr(workflow_module, "coordinator", fake_coordinator)
    monkeypatch.setattr(workflow_module, "order_agent", _parallel_node("order_agent", "order"))
    monkeypatch.setattr(workflow_module, "product_agent", _parallel_node("product_agent", "product"))
    monkeypatch.setattr(workflow_module, "summary_agent", summary_runner)
    monkeypatch.setattr(workflow_module, "chart_agent", lambda _state: {})
    monkeypatch.setattr(workflow_module, "excel_agent", lambda _state: {})

    result = workflow_module.build_graph().invoke(_build_initial_state(plan=plan))
    assert result["results"]["order"]["content"] == "order_agent done"
    assert result["results"]["product"]["content"] == "product_agent done"
    assert result["results"]["summary"]["content"] == "summary done"


def test_planner_waits_optional_dependencies_until_terminal():
    plan = [
        _make_step("s1", "order_agent"),
        _make_step("s2", "product_agent"),
        _make_step(
            "s3",
            "summary_agent",
            depends_on=["s2"],
            optional_depends_on=["s1"],
            read_from=["s1", "s2"],
            final_output=True,
        ),
    ]
    first = workflow_module.planner(_build_initial_state(plan=plan, routing={}))["routing"]
    assert set(first["next_nodes"]) == {"order_agent", "product_agent"}

    # 仅 s2 完成时，s3 还不能执行（s1 optional 还未终态），此时会继续调度 s1。
    second = workflow_module.planner(
        _build_initial_state(
            plan=plan,
            routing=first,
            step_outputs={"s2": {"status": "completed"}},
        )
    )["routing"]
    assert second["next_nodes"] == ["order_agent"]

    # s1 失败后进入终态，s3 可执行
    third = workflow_module.planner(
        _build_initial_state(
            plan=plan,
            routing=second,
            step_outputs={"s2": {"status": "completed"}, "s1": {"status": "failed"}},
        )
    )["routing"]
    assert third["next_nodes"] == ["summary_agent"]


def test_planner_routes_to_chat_when_final_step_unreachable():
    plan = [
        _make_step("s1", "order_agent"),
        _make_step("s2", "summary_agent", depends_on=["s1"], final_output=True),
    ]
    result = workflow_module.planner(
        _build_initial_state(
            plan=plan,
            step_outputs={"s1": {"status": "failed", "error": "boom"}},
        )
    )
    routing = result["routing"]
    assert routing["next_nodes"] == ["chat_agent"]
    assert routing["fallback_context"]["trigger"] == "final_output_unreachable"
