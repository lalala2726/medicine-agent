import pytest

from app.core.assistant_status import (
    reset_status_emitter,
    set_status_emitter,
    status_node,
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
        {"node": "router", "state": "start", "message": "正在分析问题"},
        {"node": "router", "state": "end"},
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
        {"node": "order", "state": "start", "message": "正在处理订单问题"},
        {
            "node": "order",
            "state": "end",
            "result": "error",
            "message": "订单节点处理失败",
        },
    ]
