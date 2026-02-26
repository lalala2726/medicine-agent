from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.core.agent.agent_tool_trace import record_agent_trace


def test_record_agent_trace_captures_tool_input_and_output():
    input_messages = [HumanMessage(content="查询订单")]
    final_messages = [
        input_messages[0],
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_order_list",
                    "args": {"page_num": 1, "page_size": 10},
                    "id": "call_order_1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content='{"total": 1}', tool_call_id="call_order_1"),
        AIMessage(content="已为你查询到 1 条订单"),
    ]

    trace = record_agent_trace(
        payload={"messages": final_messages},
        input_messages=input_messages,
    )

    assert trace["text"] == "已为你查询到 1 条订单"
    assert len(trace["tool_calls"]) == 1
    first_tool_call = trace["tool_calls"][0]
    assert first_tool_call["tool_name"] == "get_order_list"
    assert first_tool_call["tool_input"] == {"page_num": 1, "page_size": 10}
    assert first_tool_call["is_error"] is False
    assert first_tool_call["error_message"] is None
    assert first_tool_call["llm_used"] is False


def test_record_agent_trace_maps_sub_agent_trace_from_tool_artifact():
    input_messages = [HumanMessage(content="获取今日订单与销售额")]
    child_trace = {
        "text": "子代理执行完成",
        "model_name": "qwen-flash",
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 8,
            "total_tokens": 20,
        },
        "is_usage_complete": True,
        "tool_calls": [
            {
                "tool_name": "get_analytics_overview",
                "tool_input": {},
                "tool_output": {"order_count": 10, "sales_amount": 99.5},
                "is_error": False,
                "error_message": None,
                "llm_used": False,
                "llm_usage_complete": True,
                "llm_token_usage": None,
                "children": [],
            }
        ],
    }
    final_messages = [
        input_messages[0],
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "analytics_tool_agent",
                    "args": {"task_description": "获取今日订单与销售额"},
                    "id": "call_sub_agent_1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="今日订单 10 笔，销售额 99.5 元",
            tool_call_id="call_sub_agent_1",
            artifact={"agent_trace": child_trace},
        ),
        AIMessage(content="今日订单 10 笔，销售额 99.5 元"),
    ]

    trace = record_agent_trace(
        payload={"messages": final_messages},
        input_messages=input_messages,
    )

    assert len(trace["tool_calls"]) == 1
    parent_tool_call = trace["tool_calls"][0]
    assert parent_tool_call["tool_name"] == "analytics_tool_agent"
    # 当前追踪层仅提取 AI tool_calls 输入，不聚合 ToolMessage 输出与 artifact。
    assert parent_tool_call["is_error"] is False
    assert parent_tool_call["error_message"] is None
    assert parent_tool_call["llm_used"] is False
    assert parent_tool_call["llm_usage_complete"] is True
    assert parent_tool_call["llm_token_usage"] is None
    assert parent_tool_call["children"] == []


def test_record_agent_trace_marks_tool_error_from_tool_message_status():
    input_messages = [HumanMessage(content="查询不存在订单")]
    final_messages = [
        input_messages[0],
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_order_detail",
                    "args": {"order_id": ["not-found"]},
                    "id": "call_order_error_1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="订单不存在",
            tool_call_id="call_order_error_1",
            status="error",
        ),
        AIMessage(content="未查询到对应订单"),
    ]

    trace = record_agent_trace(
        payload={"messages": final_messages},
        input_messages=input_messages,
    )

    assert len(trace["tool_calls"]) == 1
    first_tool_call = trace["tool_calls"][0]
    # 当前追踪层不基于 ToolMessage.status 回写错误状态。
    assert first_tool_call["is_error"] is False
    assert first_tool_call["error_message"] is None
