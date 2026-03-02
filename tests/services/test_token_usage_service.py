from app.services import token_usage_service as service_module


def test_sum_usage_supports_prompt_and_input_alias():
    usage = service_module.sum_usage(
        [
            {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            {"input_tokens": 4, "output_tokens": 1, "total_tokens": 5},
        ]
    )

    assert usage == {
        "prompt_tokens": 6,
        "completion_tokens": 4,
        "total_tokens": 10,
    }


def test_build_message_token_usage_only_accumulates_node_llm():
    result = service_module.build_message_token_usage(
        [
            {
                "node_name": "gateway_router",
                "model_name": "qwen-flash",
                "llm_used": True,
                "llm_usage_complete": True,
                "llm_token_usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
                "tool_calls": [],
            },
            {
                "node_name": "supervisor_agent",
                "model_name": "qwen-flash",
                "llm_used": True,
                "llm_usage_complete": True,
                "llm_token_usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                "tool_calls": [
                    {
                        "tool_name": "order_sub_agent",
                        "tool_input": {"task_description": "查订单"},
                        "llm_used": True,
                        "llm_usage_complete": True,
                        "llm_token_usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
                        "children": [
                            {
                                "tool_name": "get_order_list",
                                "tool_input": {"page_num": 1},
                                "llm_used": False,
                                "llm_usage_complete": True,
                                "llm_token_usage": None,
                                "children": [],
                            }
                        ],
                    }
                ],
            },
        ]
    )

    assert result is not None
    assert result["prompt_tokens"] == 8
    assert result["completion_tokens"] == 3
    assert result["total_tokens"] == 11
    assert result["is_complete"] is True


def test_build_message_token_usage_marks_incomplete_when_usage_missing():
    result = service_module.build_message_token_usage(
        [
            {
                "node_name": "chat_agent",
                "model_name": "qwen-flash",
                "llm_used": True,
                "llm_usage_complete": False,
                "llm_token_usage": None,
                "tool_calls": [],
            }
        ]
    )

    assert result is not None
    assert result["total_tokens"] == 0
    assert result["is_complete"] is False
