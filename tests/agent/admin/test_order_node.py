import app.agent.admin.order_node as order_module


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


def _build_state(routing: dict, plan: list | None = None) -> dict:
    return {
        "user_input": "查订单",
        "user_intent": {},
        "plan": plan or [],
        "routing": routing,
        "order_context": {},
        "aftersale_context": {},
        "excel_context": {},
        "shared_memory": {},
        "results": {},
        "errors": [],
    }


def test_order_agent_streams_when_gateway_routes_directly(monkeypatch):
    model = _DummyModel()
    monkeypatch.setattr(order_module, "create_chat_model", lambda *args, **kwargs: model)

    state = _build_state(
        routing={
            "route_target": "order_agent",
            "next_nodes": [],
            "is_final_stage": False,
        },
        plan=[],
    )
    result = order_module.order_agent(state)

    assert model.stream_called is True
    assert result["order_context"]["result"]["is_end"] is True
    assert result["order_context"]["result"]["content"] == "hello order"
    assert result["order_context"]["stream_chunks"] == ["hello ", "order"]


def test_order_agent_streams_when_planner_marks_final_stage(monkeypatch):
    model = _DummyModel()
    monkeypatch.setattr(order_module, "create_chat_model", lambda *args, **kwargs: model)

    state = _build_state(
        routing={
            "route_target": "coordinator_agent",
            "next_nodes": ["order_agent"],
            "is_final_stage": True,
        },
        plan=[
            {"node_name": "order_agent", "task_description": "收尾输出"},
        ],
    )
    result = order_module.order_agent(state)

    assert model.stream_called is True
    assert result["order_context"]["result"]["is_end"] is True
    assert result["order_context"]["result"]["content"] == "hello order"
    assert result["order_context"]["stream_chunks"] == ["hello ", "order"]


def test_order_agent_uses_non_stream_when_not_final(monkeypatch):
    model = _DummyModel()
    monkeypatch.setattr(order_module, "create_chat_model", lambda *args, **kwargs: model)

    state = _build_state(
        routing={
            "route_target": "coordinator_agent",
            "next_nodes": ["order_agent"],
            "is_final_stage": False,
        },
        plan=[
            {"node_name": "order_agent", "task_description": "中间步骤"},
            {"node_name": "chart_agent", "task_description": "后续步骤"},
        ],
    )
    result = order_module.order_agent(state)

    assert model.stream_called is False
    assert model.invoke_called is True
    assert result["order_context"]["result"]["is_end"] is False
    assert result["order_context"]["result"]["content"] == "hello order"
    assert "stream_chunks" not in result["order_context"]
