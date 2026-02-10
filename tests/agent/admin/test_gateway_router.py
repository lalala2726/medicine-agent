import json

import pytest

import app.agent.admin.coordinator_node as supervisor_module
import app.agent.admin.workflow as workflow_module


class _DummyResponse:
    def __init__(self, content):
        self.content = content


class _DummyModel:
    def __init__(self, payload: str):
        self.payload = payload

    def invoke(self, _messages):
        return _DummyResponse(self.payload)


def _build_initial_state(user_input: str, plan=None, routing=None) -> dict:
    return {
        "user_input": user_input,
        "user_intent": {},
        "plan": plan or [],
        "routing": routing or {},
        "order_context": {},
        "aftersale_context": {},
        "excel_context": {},
        "shared_memory": {},
        "results": {},
        "errors": [],
    }


def test_gateway_router_routes_to_order_agent(monkeypatch: pytest.MonkeyPatch):
    payload = json.dumps(
        {
            "route_target": "order_agent",
            "difficulty": "simple",
        }
    )

    monkeypatch.setattr(
        workflow_module,
        "create_chat_model",
        lambda *args, **kwargs: _DummyModel(payload),
    )

    state = _build_initial_state("帮我查一下订单123的物流状态")
    result = workflow_module.gateway_router(state)
    routing = result["routing"]

    assert routing["route_target"] == "order_agent"
    assert routing["difficulty"] == "simple"


def test_gateway_router_routes_to_coordinator_agent(monkeypatch: pytest.MonkeyPatch):
    payload = json.dumps(
        {
            "route_target": "coordinator_agent",
            "difficulty": "complex",
        }
    )

    monkeypatch.setattr(
        workflow_module,
        "create_chat_model",
        lambda *args, **kwargs: _DummyModel(payload),
    )

    state = _build_initial_state("查异常订单并导出报表再总结建议")
    result = workflow_module.gateway_router(state)
    routing = result["routing"]

    assert routing["route_target"] == "coordinator_agent"
    assert routing["difficulty"] == "complex"


@pytest.mark.parametrize(
    ("difficulty",),
    [
        ("simple",),
        ("medium",),
        ("complex",),
        ("unknown",),
    ],
)
def test_planner_keeps_difficulty_in_valid_range(difficulty: str):
    state = _build_initial_state(
        "任意请求",
        plan=[],
        routing={
            "difficulty": difficulty,
            "stage_index": 0,
        },
    )

    assert workflow_module._normalize_difficulty(difficulty) in {"simple", "medium", "complex"}


@pytest.mark.parametrize(
    ("difficulty", "expected_model"),
    [
        ("simple", "qwen-flash"),
        ("medium", "qwen-plus"),
        ("complex", "qwen-max"),
        ("unknown", "qwen-flash"),
    ],
)
def test_coordinator_switches_model_by_difficulty(
    monkeypatch: pytest.MonkeyPatch,
    difficulty: str,
    expected_model: str,
):
    captured: dict[str, str] = {}

    class _CoordinatorModel:
        def invoke(self, _messages):
            return _DummyResponse(
                json.dumps(
                    {
                        "user_intent": {"type": "test"},
                        "plan": [],
                    }
                )
            )

    def _fake_create_chat_model(*, model: str, **kwargs):
        captured["model"] = model
        return _CoordinatorModel()

    monkeypatch.setattr(supervisor_module, "create_chat_model", _fake_create_chat_model)

    state = _build_initial_state(
        "生成计划",
        plan=[],
        routing={"difficulty": difficulty},
    )
    result = supervisor_module.coordinator(state)

    assert captured["model"] == expected_model
    assert "plan" in result


def test_planner_returns_parallel_stage_nodes_and_next_stage():
    plan = [
        [
            {"node_name": "order_agent", "task_description": "查订单"},
            {"node_name": "excel_agent", "task_description": "导出表格"},
        ],
        {"node_name": "chart_agent", "task_description": "画图"},
    ]
    state = _build_initial_state(
        "并行后汇总",
        plan=plan,
        routing={
            "difficulty": "medium",
            "stage_index": 0,
        },
    )

    first = workflow_module.planner(state)["routing"]
    assert set(first["next_nodes"]) == {"order_agent", "excel_agent"}
    assert first["stage_index"] == 1

    second_state = _build_initial_state(
        "并行后汇总",
        plan=plan,
        routing=first,
    )
    second = workflow_module.planner(second_state)["routing"]
    assert second["next_nodes"] == ["chart_agent"]
    assert second["stage_index"] == 2
