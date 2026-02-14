import json

import pytest

import app.agent.admin.dag_rules as dag_rules_module
import app.agent.admin.node.coordinator_node as coordinator_module
import app.agent.admin.workflow as workflow_module


class _DummyResponse:
    def __init__(self, content):
        self.content = content


class _DummyModel:
    def __init__(self, payload: str):
        self.payload = payload

    def invoke(self, _messages):
        return _DummyResponse(self.payload)


def _build_initial_state(user_input: str, plan=None, routing=None, step_outputs=None) -> dict:
    return {
        "user_input": user_input,
        "user_intent": {},
        "plan": plan or [],
        "routing": routing or {},
        "order_context": {},
        "product_context": {},
        "aftersale_context": {},
        "excel_context": {},
        "history_messages": [],
        "step_outputs": step_outputs or {},
        "shared_memory": {},
        "results": {},
        "errors": [],
    }


def test_gateway_router_routes_to_order_agent(monkeypatch: pytest.MonkeyPatch):
    payload = json.dumps({"route_target": "order_agent", "difficulty": "simple"})
    monkeypatch.setattr(
        workflow_module,
        "create_chat_model",
        lambda *args, **kwargs: _DummyModel(payload),
    )

    result = workflow_module.gateway_router(_build_initial_state("帮我查订单"))
    assert result["routing"]["route_target"] == "order_agent"
    assert result["routing"]["difficulty"] == "simple"


def test_gateway_router_routes_to_coordinator_agent(monkeypatch: pytest.MonkeyPatch):
    payload = json.dumps({"route_target": "coordinator_agent", "difficulty": "complex"})
    monkeypatch.setattr(
        workflow_module,
        "create_chat_model",
        lambda *args, **kwargs: _DummyModel(payload),
    )

    result = workflow_module.gateway_router(_build_initial_state("查订单并出图再总结"))
    assert result["routing"]["route_target"] == "coordinator_agent"
    assert result["routing"]["difficulty"] == "complex"


@pytest.mark.parametrize("difficulty", ["simple", "medium", "complex", "unknown"])
def test_planner_keeps_difficulty_in_valid_range(difficulty: str):
    assert workflow_module._normalize_difficulty(difficulty) in {
        "simple",
        "medium",
        "complex",
    }


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
        monkeypatch: pytest.MonkeyPatch, difficulty: str, expected_model: str
):
    captured: dict[str, str] = {}

    class _CoordinatorModel:
        def invoke(self, _messages):
            return _DummyResponse(json.dumps({"plan": []}))

    def _fake_create_chat_model(*, model: str, **_kwargs):
        captured["model"] = model
        return _CoordinatorModel()

    monkeypatch.setattr(coordinator_module, "create_chat_model", _fake_create_chat_model)
    coordinator_module.coordinator(_build_initial_state("生成计划", routing={"difficulty": difficulty}))

    assert captured["model"] == expected_model


def _valid_plan() -> list[dict]:
    return [
        {
            "step_id": "s1",
            "node_name": "order_agent",
            "task_description": "查询订单",
            "depends_on": [],
            "read_from": [],
            "include_user_input": False,
            "include_chat_history": False,
            "final_output": False,
        },
        {
            "step_id": "s2",
            "node_name": "summary_agent",
            "task_description": "输出结论",
            "depends_on": ["s1"],
            "read_from": ["s1"],
            "include_user_input": False,
            "include_chat_history": False,
            "final_output": True,
        },
    ]


def test_review_plan_accepts_valid_dag():
    is_valid, normalized, reason = dag_rules_module.review_plan(_valid_plan(), "medium")
    assert is_valid is True
    assert reason == "ok"
    assert len(normalized) == 2
    assert normalized[1]["final_output"] is True


def test_review_plan_rejects_coordinator_node_name():
    bad = _valid_plan()
    bad[0]["node_name"] = "coordinator_agent"
    is_valid, normalized, reason = dag_rules_module.review_plan(bad, "medium")
    assert is_valid is False
    assert normalized == []
    assert "coordinator_agent" in reason


def test_review_plan_rejects_cycle():
    bad = _valid_plan()
    bad[0]["depends_on"] = ["s2"]
    is_valid, _, reason = dag_rules_module.review_plan(bad, "medium")
    assert is_valid is False
    assert "循环依赖" in reason


def test_review_plan_rejects_missing_dependency():
    bad = _valid_plan()
    bad[1]["depends_on"] = ["missing"]
    is_valid, _, reason = dag_rules_module.review_plan(bad, "medium")
    assert is_valid is False
    assert "不存在" in reason


def test_review_plan_rejects_illegal_read_from():
    plan = [
        {
            "step_id": "s1",
            "node_name": "order_agent",
            "task_description": "查询订单",
            "depends_on": [],
            "read_from": [],
            "include_user_input": False,
            "include_chat_history": False,
            "final_output": False,
        },
        {
            "step_id": "s2",
            "node_name": "product_agent",
            "task_description": "查商品",
            "depends_on": [],
            "read_from": ["s1"],
            "include_user_input": False,
            "include_chat_history": False,
            "final_output": True,
        },
    ]
    is_valid, _, reason = dag_rules_module.review_plan(plan, "medium")
    assert is_valid is False
    assert "不可达上游" in reason


def test_review_plan_requires_single_final_output():
    bad = _valid_plan()
    bad[0]["final_output"] = True
    is_valid, _, reason = dag_rules_module.review_plan(bad, "medium")
    assert is_valid is False
    assert "仅能出现一次" in reason


def test_review_plan_rejects_final_output_as_dependency():
    bad = _valid_plan()
    bad.append(
        {
            "step_id": "s3",
            "node_name": "chart_agent",
            "task_description": "画图",
            "depends_on": ["s2"],
            "read_from": ["s2"],
            "include_user_input": False,
            "include_chat_history": False,
            "final_output": False,
        }
    )
    is_valid, _, reason = dag_rules_module.review_plan(bad, "complex")
    assert is_valid is False
    assert "不能被其他步骤依赖" in reason


def test_coordinator_retries_when_plan_review_fails(monkeypatch: pytest.MonkeyPatch):
    call_count = {"value": 0}
    invalid_payload = json.dumps(
        {
            "plan": [
                {
                    "step_id": "s1",
                    "node_name": "coordinator_agent",
                    "task_description": "bad",
                    "depends_on": [],
                    "read_from": [],
                    "include_user_input": False,
                    "include_chat_history": False,
                    "final_output": True,
                }
            ]
        }
    )
    valid_payload = json.dumps({"plan": _valid_plan()})

    class _RetryModel:
        def invoke(self, _messages):
            call_count["value"] += 1
            return _DummyResponse(invalid_payload if call_count["value"] == 1 else valid_payload)

    monkeypatch.setattr(coordinator_module, "create_chat_model", lambda *args, **kwargs: _RetryModel())
    result = coordinator_module.coordinator(_build_initial_state("先查后总结"))

    assert call_count["value"] == 2
    assert len(result["plan"]) == 2
    assert result["plan"][1]["final_output"] is True


def test_planner_returns_ready_nodes_with_parallel_support():
    plan = [
        {
            "step_id": "s1",
            "node_name": "order_agent",
            "task_description": "查订单",
            "depends_on": [],
            "read_from": [],
            "include_user_input": False,
            "include_chat_history": False,
            "final_output": False,
        },
        {
            "step_id": "s2",
            "node_name": "product_agent",
            "task_description": "查商品",
            "depends_on": [],
            "read_from": [],
            "include_user_input": False,
            "include_chat_history": False,
            "final_output": False,
        },
        {
            "step_id": "s3",
            "node_name": "summary_agent",
            "task_description": "汇总",
            "depends_on": ["s1", "s2"],
            "read_from": ["s1", "s2"],
            "include_user_input": False,
            "include_chat_history": False,
            "final_output": True,
        },
    ]
    first = workflow_module.planner(_build_initial_state("并行", plan=plan, routing={}))["routing"]
    assert set(first["next_nodes"]) == {"order_agent", "product_agent"}
    assert set(first["next_step_ids"]) == {"s1", "s2"}

    second = workflow_module.planner(
        _build_initial_state(
            "并行",
            plan=plan,
            routing=first,
            step_outputs={
                "s1": {"status": "completed"},
                "s2": {"status": "completed"},
            },
        )
    )["routing"]
    assert second["next_nodes"] == ["summary_agent"]
    assert second["next_step_ids"] == ["s3"]
    assert second["is_final_stage"] is True


def test_planner_blocks_downstream_when_dependency_failed():
    plan = [
        {
            "step_id": "s1",
            "node_name": "order_agent",
            "task_description": "查订单",
            "depends_on": [],
            "read_from": [],
            "include_user_input": False,
            "include_chat_history": False,
            "final_output": False,
        },
        {
            "step_id": "s2",
            "node_name": "summary_agent",
            "task_description": "汇总",
            "depends_on": ["s1"],
            "read_from": ["s1"],
            "include_user_input": False,
            "include_chat_history": False,
            "final_output": True,
        },
    ]
    result = workflow_module.planner(
        _build_initial_state(
            "失败阻断",
            plan=plan,
            step_outputs={"s1": {"status": "failed"}},
        )
    )
    routing = result["routing"]
    assert routing["next_nodes"] == []
    assert "s2" in routing["blocked_step_ids"]
    assert result["step_outputs"]["s2"]["status"] == "skipped"
