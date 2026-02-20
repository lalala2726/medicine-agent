import pytest

import app.agent.admin.workflow as workflow_module


def _build_initial_state() -> dict:
    return {
        "user_input": "test",
        "messages": [],
        "context": {},
        "routing": {},
        "results": {},
        "execution_traces": [],
        "errors": [],
    }


def test_workflow_fast_lane_order_goes_to_end(monkeypatch: pytest.MonkeyPatch):
    trace: list[str] = []

    def fake_gateway(_state: dict) -> dict:
        return {
            "routing": {
                "route_target": "order_agent",
                "target_node": "order_agent",
                "mode": "fast_lane",
            },
        }

    def fake_order(_state: dict) -> dict:
        trace.append("order_agent")
        return {"results": {"order": {"content": "ok"}}}

    monkeypatch.setattr(workflow_module, "gateway_router", fake_gateway)
    monkeypatch.setattr(workflow_module, "order_agent", fake_order)
    monkeypatch.setattr(workflow_module, "product_agent", lambda _state: {})
    monkeypatch.setattr(workflow_module, "excel_agent", lambda _state: {})
    monkeypatch.setattr(workflow_module, "chat_agent", lambda _state: {})
    monkeypatch.setattr(workflow_module, "supervisor_agent", lambda _state: {})

    graph = workflow_module.build_graph()
    result = graph.invoke(_build_initial_state())
    assert trace == ["order_agent"]
    assert result["results"]["order"]["content"] == "ok"


def test_workflow_supervisor_loop_routes_worker_then_finish(monkeypatch: pytest.MonkeyPatch):
    trace: list[str] = []

    def fake_gateway(_state: dict) -> dict:
        return {
            "routing": {
                "route_target": "supervisor_agent",
                "target_node": "supervisor_agent",
                "mode": "supervisor_loop",
                "turn": 0,
            },
        }

    def fake_supervisor(state: dict) -> dict:
        turn = int((state.get("routing") or {}).get("turn") or 0)
        if turn == 0:
            return {
                "routing": {"mode": "supervisor_loop", "turn": 1, "target_node": "product_agent"},
            }
        return {
            "routing": {"mode": "supervisor_loop", "turn": 2, "finished": True, "target_node": "summary_agent"},
        }

    def fake_product(_state: dict) -> dict:
        trace.append("product_agent")
        return {"results": {"product": {"content": "done"}}}

    def fake_summary(_state: dict) -> dict:
        trace.append("summary_agent")
        return {"results": {"summary": {"content": "final"}}}

    monkeypatch.setattr(workflow_module, "gateway_router", fake_gateway)
    monkeypatch.setattr(workflow_module, "supervisor_agent", fake_supervisor)
    monkeypatch.setattr(workflow_module, "product_agent", fake_product)
    monkeypatch.setattr(workflow_module, "order_agent", lambda _state: {})
    monkeypatch.setattr(workflow_module, "excel_agent", lambda _state: {})
    monkeypatch.setattr(workflow_module, "chat_agent", lambda _state: {})
    monkeypatch.setattr(workflow_module, "summary_agent", fake_summary)

    graph = workflow_module.build_graph()
    result = graph.invoke(_build_initial_state())
    assert trace == ["product_agent", "summary_agent"]
    assert result["results"]["product"]["content"] == "done"
    assert result["results"]["summary"]["content"] == "final"


def test_workflow_supervisor_can_route_excel_and_loop_back(monkeypatch: pytest.MonkeyPatch):
    trace: list[str] = []

    def fake_gateway(_state: dict) -> dict:
        return {
            "routing": {
                "route_target": "supervisor_agent",
                "target_node": "supervisor_agent",
                "mode": "supervisor_loop",
                "turn": 0,
            },
        }

    def fake_supervisor(state: dict) -> dict:
        turn = int((state.get("routing") or {}).get("turn") or 0)
        if turn == 0:
            return {
                "routing": {"mode": "supervisor_loop", "turn": 1, "target_node": "excel_agent"},
            }
        return {
            "routing": {"mode": "supervisor_loop", "turn": 2, "finished": True, "target_node": "summary_agent"},
        }

    def fake_excel(_state: dict) -> dict:
        trace.append("excel_agent")
        return {"results": {"excel": {"content": "excel fail"}}}

    def fake_summary(_state: dict) -> dict:
        trace.append("summary_agent")
        return {"results": {"summary": {"content": "final"}}}

    monkeypatch.setattr(workflow_module, "gateway_router", fake_gateway)
    monkeypatch.setattr(workflow_module, "supervisor_agent", fake_supervisor)
    monkeypatch.setattr(workflow_module, "excel_agent", fake_excel)
    monkeypatch.setattr(workflow_module, "product_agent", lambda _state: {})
    monkeypatch.setattr(workflow_module, "order_agent", lambda _state: {})
    monkeypatch.setattr(workflow_module, "chat_agent", lambda _state: {})
    monkeypatch.setattr(workflow_module, "summary_agent", fake_summary)

    graph = workflow_module.build_graph()
    result = graph.invoke(_build_initial_state())
    assert trace == ["excel_agent", "summary_agent"]
    assert result["results"]["excel"]["content"] == "excel fail"
    assert result["results"]["summary"]["content"] == "final"
