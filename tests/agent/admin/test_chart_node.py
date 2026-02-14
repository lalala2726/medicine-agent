import json

import app.agent.admin.node.chart_node as chart_node_module
from app.core.assistant_status import reset_status_emitter, set_status_emitter


def _build_step(
        *,
        final_output: bool = True,
        include_user_input: bool = False,
        include_chat_history: bool = False,
) -> dict:
    return {
        "step_id": "s1",
        "node_name": "chart_agent",
        "task_description": "生成订单趋势图",
        "required_depends_on": ["s0"],
        "optional_depends_on": [],
        "read_from": ["s0"],
        "include_user_input": include_user_input,
        "include_chat_history": include_chat_history,
        "final_output": final_output,
    }


def _build_state(
        *,
        route_target: str = "coordinator_agent",
        step: dict | None = None,
        step_outputs: dict | None = None,
) -> dict:
    step = step or _build_step()
    return {
        "user_input": "帮我把订单趋势画成图",
        "user_intent": {},
        "plan": [step],
        "routing": {
            "route_target": route_target,
            "next_nodes": ["chart_agent"],
            "current_step_map": {"chart_agent": step},
        },
        "order_context": {"result": {"content": "订单趋势数据"}},
        "product_context": {},
        "aftersale_context": {},
        "excel_context": {},
        "history_messages": [{"role": "assistant", "content": "上一轮建议"}],
        "step_outputs": step_outputs
        or {
            "s0": {
                "step_id": "s0",
                "node_name": "order_agent",
                "status": "completed",
                "text": "订单数据",
                "output": {"rows": [1, 2]},
            }
        },
        "shared_memory": {},
        "results": {"order": {"content": "订单结果"}},
        "errors": [],
    }


def test_chart_agent_uses_chart_tools_and_writes_result(monkeypatch):
    captured: dict = {}

    def fake_invoke_with_policy(_llm, messages, *, tools=None, **_kwargs):
        captured["tools"] = tools
        captured["messages"] = messages
        return '{"type":"line","data":[{"time":"2024-01","value":10}]}', {}

    monkeypatch.setattr(chart_node_module, "create_chat_model", lambda **_kwargs: object())
    monkeypatch.setattr(chart_node_module, "invoke_with_policy", fake_invoke_with_policy)

    result = chart_node_module.chart_agent(_build_state())
    assert captured["tools"] == chart_node_module.CHART_TOOLS
    assert result["results"]["chart"]["content"].startswith('{"type":"line"')
    assert result["results"]["chart"]["is_end"] is True
    assert result["step_outputs"]["s1"]["status"] == "completed"

    human_payload = json.loads(captured["messages"][1].content)
    assert human_payload["task_description"] == "生成订单趋势图"
    assert "upstream_outputs" in human_payload
    assert "user_input" not in human_payload
    assert "history_messages" not in human_payload


def test_chart_agent_can_include_user_and_history_when_enabled(monkeypatch):
    captured: dict = {}

    def fake_invoke_with_policy(_llm, messages, *, tools=None, **_kwargs):
        captured["messages"] = messages
        return '{"type":"line","data":[]}', {}

    monkeypatch.setattr(chart_node_module, "create_chat_model", lambda **_kwargs: object())
    monkeypatch.setattr(chart_node_module, "invoke_with_policy", fake_invoke_with_policy)

    step = _build_step(include_user_input=True, include_chat_history=True)
    chart_node_module.chart_agent(_build_state(step=step))

    human_payload = json.loads(captured["messages"][1].content)
    assert human_payload["user_input"] == "帮我把订单趋势画成图"
    assert human_payload["history_messages"][0]["role"] == "assistant"


def test_chart_agent_returns_fallback_and_failed_step_output_when_llm_fails(monkeypatch):
    def fake_invoke_with_policy(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(chart_node_module, "create_chat_model", lambda **_kwargs: object())
    monkeypatch.setattr(chart_node_module, "invoke_with_policy", fake_invoke_with_policy)

    result = chart_node_module.chart_agent(_build_state())
    assert result["results"]["chart"]["content"] == "图表服务暂时不可用，请稍后重试。"
    assert result["step_outputs"]["s1"]["status"] == "failed"


def test_chart_agent_marks_non_final_stage_is_end_false(monkeypatch):
    def fake_invoke_with_policy(_llm, _messages, *, tools=None, **_kwargs):
        assert tools == chart_node_module.CHART_TOOLS
        return '{"type":"column","data":[{"category":"A","value":1}]}', {}

    monkeypatch.setattr(chart_node_module, "create_chat_model", lambda **_kwargs: object())
    monkeypatch.setattr(chart_node_module, "invoke_with_policy", fake_invoke_with_policy)

    step = _build_step(final_output=False)
    result = chart_node_module.chart_agent(_build_state(step=step))
    assert result["results"]["chart"]["is_end"] is False


def test_chart_agent_status_hidden_when_route_not_coordinator(monkeypatch):
    def fake_invoke_with_policy(_llm, _messages, *, tools=None, **_kwargs):
        assert tools == chart_node_module.CHART_TOOLS
        return '{"type":"line","data":[]}', {}

    monkeypatch.setattr(chart_node_module, "create_chat_model", lambda **_kwargs: object())
    monkeypatch.setattr(chart_node_module, "invoke_with_policy", fake_invoke_with_policy)

    events: list[dict] = []
    token = set_status_emitter(events.append)
    try:
        chart_node_module.chart_agent(_build_state(route_target="chart_agent"))
    finally:
        reset_status_emitter(token)

    status_events = [item for item in events if item.get("type") == "status"]
    assert status_events == []


def test_chart_agent_status_visible_when_route_is_coordinator(monkeypatch):
    def fake_invoke_with_policy(_llm, _messages, *, tools=None, **_kwargs):
        assert tools == chart_node_module.CHART_TOOLS
        return '{"type":"line","data":[]}', {}

    monkeypatch.setattr(chart_node_module, "create_chat_model", lambda **_kwargs: object())
    monkeypatch.setattr(chart_node_module, "invoke_with_policy", fake_invoke_with_policy)

    events: list[dict] = []
    token = set_status_emitter(events.append)
    try:
        chart_node_module.chart_agent(_build_state(route_target="coordinator_agent"))
    finally:
        reset_status_emitter(token)

    status_events = [item for item in events if item.get("type") == "status"]
    assert status_events == [
        {
            "type": "status",
            "content": {"node": "chart", "state": "start", "message": "正在分析数据准备生成图表"},
        },
        {
            "type": "status",
            "content": {"node": "chart", "state": "end"},
        },
    ]


def test_chart_agent_marks_failed_when_model_returns_error_marker(monkeypatch):
    monkeypatch.setattr(chart_node_module, "create_chat_model", lambda **_kwargs: object())
    monkeypatch.setattr(
        chart_node_module,
        "invoke_with_policy",
        lambda *_args, **_kwargs: ("__ERROR__: 图表数据不足", {}),
    )
    result = chart_node_module.chart_agent(_build_state())
    assert result["step_outputs"]["s1"]["status"] == "failed"
    assert result["step_outputs"]["s1"]["error"] == "图表数据不足"


def test_chart_agent_marks_failed_when_tool_threshold_hit(monkeypatch):
    monkeypatch.setattr(chart_node_module, "create_chat_model", lambda **_kwargs: object())
    monkeypatch.setattr(
        chart_node_module,
        "invoke_with_policy",
        lambda *_args, **_kwargs: (
            "工具失败达到阈值",
            {"threshold_hit": True, "threshold_reason": "工具失败达到阈值"},
        ),
    )
    result = chart_node_module.chart_agent(_build_state())
    assert result["step_outputs"]["s1"]["status"] == "failed"
    assert result["step_outputs"]["s1"]["error"] == "工具失败达到阈值"
