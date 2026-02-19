from langchain_core.messages import AIMessage, HumanMessage

import app.agent.admin.node.runtime_context as runtime_module


def _build_state(
        *,
        route_target: str,
        history_messages: list[HumanMessage | AIMessage] | None = None,
        step: dict | None = None,
) -> dict:
    routing: dict = {"route_target": route_target}
    if step is not None:
        routing["current_step_map"] = {"order_agent": step}

    return {
        "user_input": "本轮用户问题",
        "user_intent": {},
        "plan": [step] if step else [],
        "routing": routing,
        "history_messages": history_messages or [],
        "step_outputs": {},
        "shared_memory": {},
        "results": {},
        "errors": [],
    }


def test_build_step_runtime_direct_mode_reads_history_by_default():
    runtime = runtime_module.build_step_runtime(
        _build_state(
            route_target="order_agent",
            history_messages=[
                HumanMessage(content="上轮问题"),
                AIMessage(content="上轮回答"),
            ],
        ),
        "order_agent",
        default_task_description="处理订单问题",
    )

    assert runtime["coordinator_mode"] is False
    assert runtime["include_chat_history"] is True
    assert runtime["include_user_input"] is False
    assert len(runtime["history_messages"]) == 2
    assert runtime["history_messages_serialized"] == [
        {"role": "user", "content": "上轮问题"},
        {"role": "assistant", "content": "上轮回答"},
    ]
    assert runtime["user_input"] == ""


def test_build_step_runtime_direct_mode_keeps_user_input_when_no_history():
    runtime = runtime_module.build_step_runtime(
        _build_state(route_target="order_agent"),
        "order_agent",
        default_task_description="处理订单问题",
    )

    assert runtime["coordinator_mode"] is False
    assert runtime["include_chat_history"] is True
    assert runtime["include_user_input"] is True
    assert runtime["history_messages"] == []
    assert runtime["history_messages_serialized"] == []
    assert runtime["user_input"] == "本轮用户问题"


def test_build_step_runtime_coordinator_mode_respects_false_flags():
    step = {
        "step_id": "s1",
        "node_name": "order_agent",
        "task_description": "查询订单",
        "required_depends_on": [],
        "optional_depends_on": [],
        "read_from": [],
        "include_user_input": False,
        "include_chat_history": False,
        "final_output": False,
    }
    runtime = runtime_module.build_step_runtime(
        _build_state(
            route_target="coordinator_agent",
            history_messages=[
                HumanMessage(content="上轮问题"),
                AIMessage(content="上轮回答"),
            ],
            step=step,
        ),
        "order_agent",
        default_task_description="处理订单问题",
    )

    assert runtime["coordinator_mode"] is True
    assert runtime["include_chat_history"] is False
    assert runtime["include_user_input"] is False
    assert runtime["history_messages"] == []
    assert runtime["history_messages_serialized"] == []
    assert runtime["user_input"] == ""


def test_build_step_runtime_coordinator_mode_respects_true_flags():
    step = {
        "step_id": "s1",
        "node_name": "order_agent",
        "task_description": "查询订单",
        "required_depends_on": [],
        "optional_depends_on": [],
        "read_from": [],
        "include_user_input": True,
        "include_chat_history": True,
        "final_output": False,
    }
    runtime = runtime_module.build_step_runtime(
        _build_state(
            route_target="coordinator_agent",
            history_messages=[
                HumanMessage(content="上轮问题"),
                AIMessage(content="上轮回答"),
            ],
            step=step,
        ),
        "order_agent",
        default_task_description="处理订单问题",
    )

    assert runtime["coordinator_mode"] is True
    assert runtime["include_chat_history"] is True
    assert runtime["include_user_input"] is True
    assert len(runtime["history_messages"]) == 2
    assert runtime["history_messages_serialized"] == [
        {"role": "user", "content": "上轮问题"},
        {"role": "assistant", "content": "上轮回答"},
    ]
    assert runtime["user_input"] == "本轮用户问题"
