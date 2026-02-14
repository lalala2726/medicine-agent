import app.agent.admin.node.product_node as product_module
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
        yield _DummyChunk("product")

    def invoke(self, _messages):
        self.invoke_called = True
        return _DummyResponse("hello product")


def _build_state(routing: dict, plan: list | None = None) -> dict:
    return {
        "user_input": "查商品",
        "user_intent": {},
        "plan": plan or [],
        "routing": routing,
        "order_context": {},
        "product_context": {},
        "aftersale_context": {},
        "excel_context": {},
        "shared_memory": {},
        "results": {},
        "errors": [],
    }


def test_product_agent_streams_when_gateway_routes_directly(monkeypatch):
    model = _DummyModel()
    monkeypatch.setattr(product_module, "create_chat_model", lambda *args, **kwargs: model)

    state = _build_state(
        routing={
            "route_target": "product_agent",
            "next_nodes": [],
            "is_final_stage": False,
        },
        plan=[],
    )
    result = product_module.product_agent(state)

    assert model.stream_called is True
    assert result["product_context"]["result"]["is_end"] is True
    assert result["product_context"]["result"]["content"] == "hello product"
    assert result["product_context"]["stream_chunks"] == ["hello ", "product"]


def test_product_agent_streams_when_planner_marks_final_stage(monkeypatch):
    model = _DummyModel()
    monkeypatch.setattr(product_module, "create_chat_model", lambda *args, **kwargs: model)

    state = _build_state(
        routing={
            "route_target": "coordinator_agent",
            "next_nodes": ["product_agent"],
            "is_final_stage": True,
        },
        plan=[
            {"node_name": "product_agent", "task_description": "收尾输出"},
        ],
    )
    result = product_module.product_agent(state)

    assert model.stream_called is True
    assert result["product_context"]["result"]["is_end"] is True
    assert result["product_context"]["result"]["content"] == "hello product"
    assert result["product_context"]["stream_chunks"] == ["hello ", "product"]


def test_product_agent_uses_non_stream_when_not_final(monkeypatch):
    model = _DummyModel()
    monkeypatch.setattr(product_module, "create_chat_model", lambda *args, **kwargs: model)

    state = _build_state(
        routing={
            "route_target": "coordinator_agent",
            "next_nodes": ["product_agent"],
            "is_final_stage": False,
        },
        plan=[
            {"node_name": "product_agent", "task_description": "中间步骤"},
            {"node_name": "chart_agent", "task_description": "后续步骤"},
        ],
    )
    result = product_module.product_agent(state)

    assert model.stream_called is False
    assert model.invoke_called is True
    assert result["product_context"]["result"]["is_end"] is False
    assert result["product_context"]["result"]["content"] == "hello product"
    assert "stream_chunks" not in result["product_context"]


def test_product_agent_status_hidden_when_route_not_coordinator(monkeypatch):
    model = _DummyModel()
    monkeypatch.setattr(product_module, "create_chat_model", lambda *args, **kwargs: model)

    events: list[dict] = []
    token = set_status_emitter(events.append)
    try:
        state = _build_state(
            routing={
                "route_target": "product_agent",
                "next_nodes": [],
                "is_final_stage": False,
            },
            plan=[],
        )
        product_module.product_agent(state)
    finally:
        reset_status_emitter(token)

    status_events = [item for item in events if item.get("type") == "status"]
    assert status_events == []


def test_product_agent_status_visible_when_route_is_coordinator(monkeypatch):
    model = _DummyModel()
    monkeypatch.setattr(product_module, "create_chat_model", lambda *args, **kwargs: model)

    events: list[dict] = []
    token = set_status_emitter(events.append)
    try:
        state = _build_state(
            routing={
                "route_target": "coordinator_agent",
                "next_nodes": ["product_agent"],
                "is_final_stage": False,
            },
            plan=[
                {"node_name": "product_agent", "task_description": "中间步骤"},
            ],
        )
        product_module.product_agent(state)
    finally:
        reset_status_emitter(token)

    status_events = [item for item in events if item.get("type") == "status"]
    assert status_events == [
        {
            "type": "status",
            "content": {"node": "product", "state": "start", "message": "正在处理商品问题"},
        },
        {
            "type": "status",
            "content": {"node": "product", "state": "end"},
        },
    ]
