import json

import app.utils.streaming_utils as streaming_utils_module
from app.core.assistant_status import reset_status_emitter, set_status_emitter


class _SuccessTool:
    async def ainvoke(self, _args: dict) -> dict:
        return {"items": [1, 2, 3]}


class _FailTool:
    async def ainvoke(self, _args: dict) -> dict:
        raise RuntimeError("boom")


def test_exec_tool_emits_status_for_success():
    events: list[dict] = []
    token = set_status_emitter(events.append)

    result = streaming_utils_module._exec_tool(
        tool_call={"name": "get_order_list", "args": {"page_num": 1}, "id": "1"},
        tool_map={"get_order_list": _SuccessTool()},
        log_enabled=False,
    )
    reset_status_emitter(token)

    assert json.loads(result) == {"items": [1, 2, 3]}
    assert events == [
        {
            "node": "tool:get_order_list",
            "state": "start",
            "message": "正在查询订单信息",
        },
        {
            "node": "tool:get_order_list",
            "state": "end",
        },
    ]


def test_exec_tool_emits_single_error_end_status():
    events: list[dict] = []
    token = set_status_emitter(events.append)

    result = streaming_utils_module._exec_tool(
        tool_call={"name": "get_order_list", "args": {"page_num": 1}, "id": "1"},
        tool_map={"get_order_list": _FailTool()},
        log_enabled=False,
    )
    reset_status_emitter(token)

    assert "工具执行失败: get_order_list" in json.loads(result)["error"]
    assert events == [
        {
            "node": "tool:get_order_list",
            "state": "start",
            "message": "正在查询订单信息",
        },
        {
            "node": "tool:get_order_list",
            "state": "end",
            "result": "error",
            "message": "订单服务调用失败",
        },
    ]
