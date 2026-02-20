import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import app.agent.admin.dag_rules as dag_rules_module
import app.agent.admin.node.coordinator_node as coordinator_module
import app.agent.tools.coordinator_tools as coordinator_tools_module
import app.agent.admin.workflow as workflow_module


class _DummyResponse:
    def __init__(self, content):
        self.content = content


class _DummyModel:
    def __init__(self, payload: str):
        self.payload = payload

    def invoke(self, _messages):
        return _DummyResponse(self.payload)


def _build_initial_state(
        user_input: str,
        plan=None,
        routing=None,
        step_outputs=None,
        history_messages=None,
) -> dict:
    return {
        "user_input": user_input,
        "user_intent": {},
        "plan": plan or [],
        "routing": routing or {},
        "history_messages": history_messages or [],
        "step_outputs": step_outputs or {},
        "execution_traces": [],
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


def test_gateway_router_uses_history_messages_when_present(monkeypatch: pytest.MonkeyPatch):
    payload = json.dumps({"route_target": "summary_agent", "difficulty": "simple"})
    captured: dict = {}

    class _CaptureModel:
        def invoke(self, messages):
            captured["messages"] = messages
            return _DummyResponse(payload)

    monkeypatch.setattr(
        workflow_module,
        "create_chat_model",
        lambda *args, **kwargs: _CaptureModel(),
    )

    result = workflow_module.gateway_router(
        _build_initial_state(
            "",
            history_messages=[
                HumanMessage(content="上轮问题"),
                AIMessage(content="上轮回答"),
                HumanMessage(content="本轮问题"),
            ],
        )
    )
    assert result["routing"]["route_target"] == "coordinator_agent"
    assert result["routing"]["difficulty"] == "simple"
    assert len(captured["messages"]) == 4
    assert isinstance(captured["messages"][1], HumanMessage)
    assert isinstance(captured["messages"][2], AIMessage)
    assert isinstance(captured["messages"][3], HumanMessage)


@pytest.mark.parametrize("forbidden_target", ["excel_agent", "chart_agent", "summary_agent"])
def test_gateway_router_forces_coordinator_for_non_direct_targets(
        monkeypatch: pytest.MonkeyPatch,
        forbidden_target: str,
):
    payload = json.dumps({"route_target": forbidden_target, "difficulty": "simple"})
    monkeypatch.setattr(
        workflow_module,
        "create_chat_model",
        lambda *args, **kwargs: _DummyModel(payload),
    )

    result = workflow_module.gateway_router(_build_initial_state("请导出并生成图表"))
    assert result["routing"]["route_target"] == "coordinator_agent"


@pytest.mark.parametrize("difficulty", ["simple", "medium", "complex", "unknown"])
def test_planner_keeps_difficulty_in_valid_range(difficulty: str):
    assert workflow_module._normalize_difficulty(difficulty) in {
        "simple",
        "medium",
        "complex",
    }


@pytest.mark.parametrize(
    ("difficulty", "expected_model", "expected_enable_thinking"),
    [
        ("simple", "qwen-max", False),
        ("medium", "qwen-max", True),
        ("complex", "qwen3.5-plus", True),
        ("unknown", "qwen-max", False),
    ],
)
def test_coordinator_switches_model_by_difficulty(
        monkeypatch: pytest.MonkeyPatch,
        difficulty: str,
        expected_model: str,
        expected_enable_thinking: bool,
):
    captured: dict[str, object] = {}

    class _CoordinatorModel:
        def invoke(self, _messages):
            return _DummyResponse(json.dumps({"plan": []}))

    def _fake_create_chat_model(*, model: str, **kwargs):
        captured["model"] = model
        captured["extra_body"] = kwargs.get("extra_body")
        return _CoordinatorModel()

    monkeypatch.setattr(coordinator_module, "create_chat_model", _fake_create_chat_model)
    coordinator_module.coordinator(_build_initial_state("生成计划", routing={"difficulty": difficulty}))

    assert captured["model"] == expected_model
    if expected_enable_thinking:
        assert captured["extra_body"] == {"enable_thinking": True}
    else:
        assert captured["extra_body"] is None


def _valid_plan() -> list[dict]:
    return [
        {
            "step_id": "s1",
            "node_name": "order_agent",
            "task_description": "查询订单",
            "required_depends_on": [],
            "optional_depends_on": [],
            "read_from": [],
            "include_user_input": False,
            "include_chat_history": False,
            "final_output": False,
        },
        {
            "step_id": "s2",
            "node_name": "summary_agent",
            "task_description": "输出结论",
            "required_depends_on": ["s1"],
            "optional_depends_on": [],
            "read_from": ["s1"],
            "include_user_input": False,
            "include_chat_history": False,
            "final_output": True,
        },
    ]


def test_get_agent_detail_returns_selected_agent_details():
    result = coordinator_tools_module.get_agent_detail.invoke(
        {
            "agent_names": ["order_agent", "product_agent"],
            "include_tool_parameters": True,
            "include_coordination_guide": True,
            "include_plan_examples": True,
        }
    )
    assert result["ok"] is True
    assert result["resolved_agents"] == ["order_agent", "product_agent"]
    assert "order_agent" in result["agent_details"]
    assert "available_tools" in result["agent_details"]["order_agent"]
    assert "coordination_guide" in result["agent_details"]["product_agent"]
    assert "plan_example" in result["agent_details"]["order_agent"]
    assert "tool_parameters_supported" in result


def test_get_agent_detail_returns_error_when_agent_not_supported():
    result = coordinator_tools_module.get_agent_detail.invoke(
        {
            "agent_names": ["unknown_agent"],
        }
    )
    assert result["ok"] is False
    assert "unknown_agent" in result["unsupported_agents"]
    assert "supported_agents" in result


def test_review_plan_accepts_valid_dag():
    is_valid, normalized, reason = dag_rules_module.review_plan(_valid_plan(), "medium")
    assert is_valid is True
    assert reason == "ok"
    assert len(normalized) == 2
    assert normalized[1]["final_output"] is True
    assert normalized[0]["failure_policy"]["mode"] == "hybrid"
    assert normalized[0]["failure_policy"]["error_marker_prefix"] == "__ERROR__:"


def test_review_plan_rejects_coordinator_node_name():
    bad = _valid_plan()
    bad[0]["node_name"] = "coordinator_agent"
    is_valid, normalized, reason = dag_rules_module.review_plan(bad, "medium")
    assert is_valid is False
    assert normalized == []
    assert "coordinator_agent" in reason


def test_review_plan_rejects_cycle():
    bad = _valid_plan()
    bad[0]["required_depends_on"] = ["s2"]
    is_valid, _, reason = dag_rules_module.review_plan(bad, "medium")
    assert is_valid is False
    assert "循环依赖" in reason


def test_review_plan_rejects_missing_dependency():
    bad = _valid_plan()
    bad[1]["required_depends_on"] = ["missing"]
    is_valid, _, reason = dag_rules_module.review_plan(bad, "medium")
    assert is_valid is False
    assert "不存在" in reason


def test_review_plan_rejects_old_depends_on_schema():
    bad = _valid_plan()
    bad[1].pop("required_depends_on", None)
    bad[1].pop("optional_depends_on", None)
    bad[1]["depends_on"] = ["s1"]
    is_valid, _, reason = dag_rules_module.review_plan(bad, "medium")
    assert is_valid is False
    assert "required_depends_on" in reason


def test_review_plan_rejects_overlap_between_required_and_optional():
    bad = _valid_plan()
    bad[1]["optional_depends_on"] = ["s1"]
    is_valid, _, reason = dag_rules_module.review_plan(bad, "medium")
    assert is_valid is False
    assert "存在重复依赖" in reason


def test_review_plan_rejects_missing_optional_dependency():
    bad = _valid_plan()
    bad[1]["optional_depends_on"] = ["missing"]
    is_valid, _, reason = dag_rules_module.review_plan(bad, "medium")
    assert is_valid is False
    assert "optional_depends_on 引用了不存在的步骤" in reason


def test_review_plan_rejects_illegal_read_from():
    plan = [
        {
            "step_id": "s1",
            "node_name": "order_agent",
            "task_description": "查询订单",
            "required_depends_on": [],
            "optional_depends_on": [],
            "read_from": [],
            "include_user_input": False,
            "include_chat_history": False,
            "final_output": False,
        },
        {
            "step_id": "s2",
            "node_name": "product_agent",
            "task_description": "查商品",
            "required_depends_on": [],
            "optional_depends_on": [],
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
            "required_depends_on": ["s2"],
            "optional_depends_on": [],
            "read_from": ["s2"],
            "include_user_input": False,
            "include_chat_history": False,
            "final_output": False,
        }
    )
    is_valid, _, reason = dag_rules_module.review_plan(bad, "complex")
    assert is_valid is False
    assert "不能被其他步骤依赖" in reason


def test_review_plan_accepts_valid_failure_policy_override():
    plan = _valid_plan()
    plan[0]["failure_policy"] = {
        "mode": "tool_only",
        "error_marker_prefix": "__ERROR__:",
        "tool_error_counting": "total",
        "max_tool_errors": 3,
        "strict_data_quality": False,
    }
    is_valid, normalized, reason = dag_rules_module.review_plan(plan, "medium")
    assert is_valid is True
    assert reason == "ok"
    assert normalized[0]["failure_policy"]["mode"] == "tool_only"
    assert normalized[0]["failure_policy"]["tool_error_counting"] == "total"
    assert normalized[0]["failure_policy"]["max_tool_errors"] == 3


def test_review_plan_rejects_invalid_failure_policy_mode():
    plan = _valid_plan()
    plan[0]["failure_policy"] = {"mode": "bad_mode"}
    is_valid, _, reason = dag_rules_module.review_plan(plan, "medium")
    assert is_valid is False
    assert "failure_policy.mode 非法" in reason


def test_review_plan_rejects_invalid_failure_policy_counting():
    plan = _valid_plan()
    plan[0]["failure_policy"] = {"tool_error_counting": "bad_count"}
    is_valid, _, reason = dag_rules_module.review_plan(plan, "medium")
    assert is_valid is False
    assert "failure_policy.tool_error_counting 非法" in reason


def test_review_plan_rejects_invalid_failure_policy_threshold():
    plan = _valid_plan()
    plan[0]["failure_policy"] = {"max_tool_errors": 0}
    is_valid, _, reason = dag_rules_module.review_plan(plan, "medium")
    assert is_valid is False
    assert "max_tool_errors 必须在 1..5" in reason


def test_coordinator_retries_when_plan_review_fails(monkeypatch: pytest.MonkeyPatch):
    call_count = {"value": 0}
    invalid_payload = json.dumps(
        {
            "plan": [
                {
                    "step_id": "s1",
                    "node_name": "coordinator_agent",
                    "task_description": "bad",
                    "required_depends_on": [],
                    "optional_depends_on": [],
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


def test_coordinator_uses_tool_mode_when_model_supports_bind_tools(
        monkeypatch: pytest.MonkeyPatch,
):
    class _ToolCapableResponse:
        def __init__(self, content: str):
            self.content = content
            self.tool_calls = []

    class _ToolCapableModel:
        def __init__(self):
            self.bind_tools_called = False

        def bind_tools(self, _tools):
            self.bind_tools_called = True
            return self

        def invoke(self, _messages):
            return _ToolCapableResponse(json.dumps({"plan": _valid_plan()}))

    model = _ToolCapableModel()
    monkeypatch.setattr(coordinator_module, "create_chat_model", lambda *args, **kwargs: model)

    result = coordinator_module.coordinator(_build_initial_state("先查订单后汇总"))
    assert model.bind_tools_called is True
    assert len(result["plan"]) == 2
    assert result["plan"][1]["final_output"] is True


def test_planner_returns_ready_nodes_with_parallel_support():
    plan = [
        {
            "step_id": "s1",
            "node_name": "order_agent",
            "task_description": "查订单",
            "required_depends_on": [],
            "optional_depends_on": [],
            "read_from": [],
            "include_user_input": False,
            "include_chat_history": False,
            "final_output": False,
        },
        {
            "step_id": "s2",
            "node_name": "product_agent",
            "task_description": "查商品",
            "required_depends_on": [],
            "optional_depends_on": [],
            "read_from": [],
            "include_user_input": False,
            "include_chat_history": False,
            "final_output": False,
        },
        {
            "step_id": "s3",
            "node_name": "summary_agent",
            "task_description": "汇总",
            "required_depends_on": ["s1", "s2"],
            "optional_depends_on": [],
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
            "required_depends_on": [],
            "optional_depends_on": [],
            "read_from": [],
            "include_user_input": False,
            "include_chat_history": False,
            "final_output": False,
        },
        {
            "step_id": "s2",
            "node_name": "summary_agent",
            "task_description": "汇总",
            "required_depends_on": ["s1"],
            "optional_depends_on": [],
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
    assert routing["next_nodes"] == ["chat_agent"]
    assert routing["fallback_context"]["trigger"] == "final_output_unreachable"
    assert "s2" in routing["blocked_step_ids"]
    assert result["step_outputs"]["s2"]["status"] == "skipped"
