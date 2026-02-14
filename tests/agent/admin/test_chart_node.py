import json

import app.agent.admin.node.chart_node as chart_node_module
from app.core.assistant_status import reset_status_emitter, set_status_emitter


def _build_state(
        *, is_final_stage: bool = True, route_target: str = "coordinator_agent"
) -> dict:
    return {
        "user_input": "帮我把订单趋势画成图",
        "user_intent": {},
        "plan": [{"node_name": "chart_agent", "task_description": "生成订单趋势图"}],
        "routing": {
            "route_target": route_target,
            "next_nodes": ["chart_agent"],
            "is_final_stage": is_final_stage,
            "current_step_map": {
                "chart_agent": {
                    "node_name": "chart_agent",
                    "task_description": "生成订单趋势图",
                }
            },
        },
        "order_context": {"result": {"content": "订单趋势数据"}},
        "aftersale_context": {},
        "excel_context": {},
        "shared_memory": {},
        "results": {"order": {"content": "订单结果"}},
        "errors": [],
    }


def test_chart_agent_uses_chart_tools_and_writes_result(monkeypatch):
    captured: dict = {}

    def fake_invoke(_llm, messages, *, tools=None, **_kwargs):
        captured["tools"] = tools
        captured["messages"] = messages
        return '{"type":"line","data":[{"time":"2024-01","value":10}]}'

    monkeypatch.setattr(chart_node_module, "create_chat_model", lambda **_kwargs: object())
    monkeypatch.setattr(chart_node_module, "invoke", fake_invoke)

    state = _build_state()
    result = chart_node_module.chart_agent(state)

    assert captured["tools"] == chart_node_module.CHART_TOOLS
    assert len(captured["tools"]) == 1
    assert captured["tools"][0].name == "get_chart_sample_by_name"
    assert result["results"]["chart"]["content"].startswith('{"type":"line"')
    assert result["results"]["chart"]["is_end"] is True

    system_prompt = captured["messages"][0].content
    assert "line（折线图）" in system_prompt
    assert "pie（饼图）" in system_prompt
    assert "get_chart_sample_by_name" in system_prompt

    human_payload = captured["messages"][1].content
    parsed = json.loads(human_payload)
    assert parsed["task_description"] == "生成订单趋势图"
    assert "order_context" in parsed


def test_chart_agent_returns_fallback_when_llm_fails(monkeypatch):
    def fake_invoke(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(chart_node_module, "create_chat_model", lambda **_kwargs: object())
    monkeypatch.setattr(chart_node_module, "invoke", fake_invoke)

    result = chart_node_module.chart_agent(_build_state())
    assert result["results"]["chart"]["content"] == "图表服务暂时不可用，请稍后重试。"
    assert result["results"]["chart"]["is_end"] is True


def test_chart_agent_marks_non_final_stage_is_end_false(monkeypatch):
    def fake_invoke(_llm, _messages, *, tools=None, **_kwargs):
        assert tools == chart_node_module.CHART_TOOLS
        return '{"type":"column","data":[{"category":"A","value":1}]}'

    monkeypatch.setattr(chart_node_module, "create_chat_model", lambda **_kwargs: object())
    monkeypatch.setattr(chart_node_module, "invoke", fake_invoke)

    result = chart_node_module.chart_agent(_build_state(is_final_stage=False))
    assert result["results"]["chart"]["is_end"] is False


def test_chart_agent_status_hidden_when_route_not_coordinator(monkeypatch):
    def fake_invoke(_llm, _messages, *, tools=None, **_kwargs):
        assert tools == chart_node_module.CHART_TOOLS
        return '{"type":"line","data":[]}'

    monkeypatch.setattr(chart_node_module, "create_chat_model", lambda **_kwargs: object())
    monkeypatch.setattr(chart_node_module, "invoke", fake_invoke)

    events: list[dict] = []
    token = set_status_emitter(events.append)
    try:
        chart_node_module.chart_agent(_build_state(route_target="chart_agent"))
    finally:
        reset_status_emitter(token)

    status_events = [item for item in events if item.get("type") == "status"]
    assert status_events == []


def test_chart_agent_status_visible_when_route_is_coordinator(monkeypatch):
    def fake_invoke(_llm, _messages, *, tools=None, **_kwargs):
        assert tools == chart_node_module.CHART_TOOLS
        return '{"type":"line","data":[]}'

    monkeypatch.setattr(chart_node_module, "create_chat_model", lambda **_kwargs: object())
    monkeypatch.setattr(chart_node_module, "invoke", fake_invoke)

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
