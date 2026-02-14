import app.agent.admin.node.order_node as order_module
from app.core.assistant_status import reset_status_emitter, set_status_emitter


class _DummyResponse:
    def __init__(self, content: str):
        self.content = content


class _DummyChunk:
    def __init__(self, content: str):
        self.content = content


class _DummyModel:
    def __init__(self):
        self.stream_called = False
        self.invoke_called = False
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def stream(self, _messages):
        self.stream_called = True
        yield _DummyChunk("hello ")
        yield _DummyChunk("order")

    def invoke(self, _messages):
        self.invoke_called = True
        return _DummyResponse("hello order")


def _build_step(final_output: bool) -> dict:
    return {
        "step_id": "s1",
        "node_name": "order_agent",
        "task_description": "处理订单",
        "depends_on": [],
        "read_from": [],
        "include_user_input": False,
        "include_chat_history": False,
        "final_output": final_output,
    }


def _build_state(routing: dict, plan: list | None = None) -> dict:
    return {
        "user_input": "查订单",
        "user_intent": {},
        "plan": plan or [],
        "routing": routing,
        "order_context": {},
        "product_context": {},
        "aftersale_context": {},
        "excel_context": {},
        "history_messages": [],
        "step_outputs": {},
        "shared_memory": {},
        "results": {},
        "errors": [],
    }


def test_order_agent_streams_when_gateway_routes_directly(monkeypatch):
    model = _DummyModel()
    monkeypatch.setattr(order_module, "create_chat_model", lambda *args, **kwargs: model)

    state = _build_state(routing={"route_target": "order_agent", "next_nodes": []}, plan=[])
    result = order_module.order_agent(state)

    assert model.stream_called is True
    assert result["order_context"]["result"]["is_end"] is True
    assert result["order_context"]["result"]["content"] == "hello order"
    assert result["order_context"]["stream_chunks"] == ["hello ", "order"]
    assert "step_outputs" not in result


def test_order_agent_streams_when_current_step_marked_final(monkeypatch):
    model = _DummyModel()
    monkeypatch.setattr(order_module, "create_chat_model", lambda *args, **kwargs: model)

    step = _build_step(final_output=True)
    state = _build_state(
        routing={
            "route_target": "coordinator_agent",
            "next_nodes": ["order_agent"],
            "current_step_map": {"order_agent": step},
        },
        plan=[step],
    )
    result = order_module.order_agent(state)

    assert model.stream_called is True
    assert result["order_context"]["result"]["is_end"] is True
    assert result["step_outputs"]["s1"]["status"] == "completed"
    assert result["step_outputs"]["s1"]["node_name"] == "order_agent"


def test_order_agent_uses_non_stream_when_not_final(monkeypatch):
    model = _DummyModel()
    monkeypatch.setattr(order_module, "create_chat_model", lambda *args, **kwargs: model)

    step = _build_step(final_output=False)
    state = _build_state(
        routing={
            "route_target": "coordinator_agent",
            "next_nodes": ["order_agent"],
            "current_step_map": {"order_agent": step},
        },
        plan=[step],
    )
    result = order_module.order_agent(state)

    assert model.stream_called is False
    assert model.invoke_called is True
    assert result["order_context"]["result"]["is_end"] is False
    assert result["order_context"]["result"]["content"] == "hello order"
    assert "stream_chunks" not in result["order_context"]
    assert result["step_outputs"]["s1"]["status"] == "completed"


def test_order_agent_status_hidden_when_route_not_coordinator(monkeypatch):
    model = _DummyModel()
    monkeypatch.setattr(order_module, "create_chat_model", lambda *args, **kwargs: model)

    events: list[dict] = []
    token = set_status_emitter(events.append)
    try:
        state = _build_state(routing={"route_target": "order_agent"}, plan=[])
        order_module.order_agent(state)
    finally:
        reset_status_emitter(token)

    status_events = [item for item in events if item.get("type") == "status"]
    assert status_events == []


def test_order_agent_status_visible_when_route_is_coordinator(monkeypatch):
    model = _DummyModel()
    monkeypatch.setattr(order_module, "create_chat_model", lambda *args, **kwargs: model)

    events: list[dict] = []
    token = set_status_emitter(events.append)
    try:
        step = _build_step(final_output=False)
        state = _build_state(
            routing={
                "route_target": "coordinator_agent",
                "next_nodes": ["order_agent"],
                "current_step_map": {"order_agent": step},
            },
            plan=[step],
        )
        order_module.order_agent(state)
    finally:
        reset_status_emitter(token)

    status_events = [item for item in events if item.get("type") == "status"]
    assert status_events == [
        {
            "type": "status",
            "content": {"node": "order", "state": "start", "message": "正在处理订单问题"},
        },
        {
            "type": "status",
            "content": {"node": "order", "state": "end"},
        },
    ]
