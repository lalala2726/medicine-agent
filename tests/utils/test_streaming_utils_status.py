import json

import app.utils.streaming_utils as streaming_utils_module
from app.core.assistant_status import (
    reset_status_emitter,
    set_status_emitter,
    tool_call_status,
)


class _DecoratedSuccessTool:
    @tool_call_status(tool_name="get_order_list")
    async def _runner(self, _args: dict) -> dict:
        return {"items": [1, 2, 3]}

    async def ainvoke(self, args: dict) -> dict:
        return await self._runner(args)


class _DecoratedNoReturnTool:
    @tool_call_status(tool_name="get_order_list")
    async def _runner(self, _args: dict) -> None:
        return None

    async def ainvoke(self, args: dict) -> None:
        return await self._runner(args)


def test_exec_tool_known_tool_events_are_emitted_by_decorator_only():
    events: list[dict] = []
    token = set_status_emitter(events.append)

    result = streaming_utils_module._exec_tool(
        tool_call={"name": "get_order_list", "args": {"page_num": 1}, "id": "1"},
        tool_map={"get_order_list": _DecoratedSuccessTool()},
        log_enabled=False,
    )
    reset_status_emitter(token)

    assert json.loads(result) == {"items": [1, 2, 3]}
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


def test_exec_tool_known_tool_without_result_emits_timely():
    events: list[dict] = []
    token = set_status_emitter(events.append)

    result = streaming_utils_module._exec_tool(
        tool_call={"name": "get_order_list", "args": {"page_num": 1}, "id": "1"},
        tool_map={"get_order_list": _DecoratedNoReturnTool()},
        log_enabled=False,
    )
    reset_status_emitter(token)

    assert result == "None"
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


def test_exec_tool_unknown_tool_emits_fallback_error_events():
    events: list[dict] = []
    token = set_status_emitter(events.append)

    result = streaming_utils_module._exec_tool(
        tool_call={"name": "unknown_tool", "args": {}, "id": "1"},
        tool_map={},
        log_enabled=False,
    )
    reset_status_emitter(token)

    assert json.loads(result)["error"] == "未知工具: unknown_tool"
    assert events == [
        {
            "type": "function_call",
            "content": {
                "node": "tool:unknown_tool",
                "state": "start",
                "message": "正在调用工具 unknown_tool",
            },
        },
        {
            "type": "function_call",
            "content": {
                "node": "tool:unknown_tool",
                "state": "end",
                "result": "error",
                "message": "未知工具: unknown_tool",
            },
        },
    ]
