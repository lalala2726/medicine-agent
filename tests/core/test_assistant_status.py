import asyncio

import pytest

from app.core.assistant_status import (
    reset_status_emitter,
    set_status_emitter,
    status_node,
    tool_call_status,
)


def test_status_node_emits_start_and_end():
    events: list[dict] = []
    token = set_status_emitter(events.append)

    @status_node(node="router", start_message="正在分析问题")
    def _node() -> str:
        return "ok"

    assert _node() == "ok"
    reset_status_emitter(token)

    assert events == [
        {
            "type": "status",
            "content": {"node": "router", "state": "start", "message": "正在分析问题"},
        },
        {
            "type": "status",
            "content": {"node": "router", "state": "end"},
        },
    ]


def test_status_node_emits_error_end_on_exception():
    events: list[dict] = []
    token = set_status_emitter(events.append)

    @status_node(
        node="order",
        start_message="正在处理订单问题",
        error_message="订单节点处理失败",
    )
    def _node() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        _node()
    reset_status_emitter(token)

    assert events == [
        {
            "type": "status",
            "content": {"node": "order", "state": "start", "message": "正在处理订单问题"},
        },
        {
            "type": "status",
            "content": {
                "node": "order",
                "state": "end",
                "result": "error",
                "message": "订单节点处理失败",
            },
        },
    ]


def test_status_node_after_coordinator_emits_when_route_is_coordinator():
    events: list[dict] = []
    token = set_status_emitter(events.append)

    @status_node(
        node="order",
        start_message="正在处理订单问题",
        display_when="after_coordinator",
    )
    def _node(state: dict) -> str:
        return "ok"

    state = {"routing": {"route_target": "coordinator_agent"}}
    assert _node(state) == "ok"
    reset_status_emitter(token)

    assert events == [
        {
            "type": "status",
            "content": {"node": "order", "state": "start", "message": "正在处理订单问题"},
        },
        {
            "type": "status",
            "content": {"node": "order", "state": "end"},
        },
    ]


def test_status_node_after_coordinator_does_not_emit_when_route_is_not_coordinator():
    events: list[dict] = []
    token = set_status_emitter(events.append)

    @status_node(
        node="order",
        start_message="正在处理订单问题",
        display_when="after_coordinator",
    )
    def _node(state: dict) -> str:
        return "ok"

    state = {"routing": {"route_target": "order_agent"}}
    assert _node(state) == "ok"
    reset_status_emitter(token)

    assert events == []


def test_status_node_after_coordinator_does_not_emit_when_route_target_missing():
    events: list[dict] = []
    token = set_status_emitter(events.append)

    @status_node(
        node="order",
        start_message="正在处理订单问题",
        display_when="after_coordinator",
    )
    def _node(state: dict) -> str:
        return "ok"

    state = {"routing": {}}
    assert _node(state) == "ok"
    reset_status_emitter(token)

    assert events == []


def test_status_node_never_does_not_emit_events():
    events: list[dict] = []
    token = set_status_emitter(events.append)

    @status_node(
        node="order",
        start_message="正在处理订单问题",
        display_when="never",
    )
    def _node() -> str:
        return "ok"

    assert _node() == "ok"
    reset_status_emitter(token)

    assert events == []


def test_status_node_after_coordinator_exception_emits_error_when_enabled():
    events: list[dict] = []
    token = set_status_emitter(events.append)

    @status_node(
        node="order",
        start_message="正在处理订单问题",
        error_message="订单节点处理失败",
        display_when="after_coordinator",
    )
    def _node(state: dict) -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        _node({"routing": {"route_target": "coordinator_agent"}})
    reset_status_emitter(token)

    assert events == [
        {
            "type": "status",
            "content": {"node": "order", "state": "start", "message": "正在处理订单问题"},
        },
        {
            "type": "status",
            "content": {
                "node": "order",
                "state": "end",
                "result": "error",
                "message": "订单节点处理失败",
            },
        },
    ]


def test_status_node_after_coordinator_exception_does_not_emit_when_disabled():
    events: list[dict] = []
    token = set_status_emitter(events.append)

    @status_node(
        node="order",
        start_message="正在处理订单问题",
        error_message="订单节点处理失败",
        display_when="after_coordinator",
    )
    def _node(state: dict) -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        _node({"routing": {"route_target": "order_agent"}})
    reset_status_emitter(token)

    assert events == []


def test_tool_call_status_emits_start_and_end():
    events: list[dict] = []
    token = set_status_emitter(events.append)

    @tool_call_status(tool_name="get_order_list")
    async def _tool() -> dict:
        return {"ok": True}

    assert asyncio.run(_tool()) == {"ok": True}
    reset_status_emitter(token)

    assert events == [
        {
            "type": "function_call",
            "content": {
                "node": "tool:get_order_list",
                "state": "start",
                "message": "正在查询订单信息",
            },
        },
        {
            "type": "function_call",
            "content": {
                "node": "tool:get_order_list",
                "state": "end",
            },
        },
    ]


def test_tool_call_status_emits_timely_when_result_is_none():
    events: list[dict] = []
    token = set_status_emitter(events.append)

    @tool_call_status(tool_name="get_order_list")
    async def _tool() -> None:
        return None

    assert asyncio.run(_tool()) is None
    reset_status_emitter(token)

    assert events == [
        {
            "type": "function_call",
            "content": {
                "node": "tool:get_order_list",
                "state": "start",
                "message": "正在查询订单信息",
            },
        },
        {
            "type": "function_call",
            "content": {
                "node": "tool:get_order_list",
                "state": "timely",
                "message": "订单信息正在持续处理中",
            },
        },
    ]


def test_tool_call_status_emits_error_end_on_exception():
    events: list[dict] = []
    token = set_status_emitter(events.append)

    @tool_call_status(tool_name="get_order_list")
    async def _tool() -> dict:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        asyncio.run(_tool())
    reset_status_emitter(token)

    assert events == [
        {
            "type": "function_call",
            "content": {
                "node": "tool:get_order_list",
                "state": "start",
                "message": "正在查询订单信息",
            },
        },
        {
            "type": "function_call",
            "content": {
                "node": "tool:get_order_list",
                "state": "end",
                "result": "error",
                "message": "订单服务调用失败",
            },
        },
    ]
