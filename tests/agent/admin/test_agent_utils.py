import app.agent.admin.agent_utils as agent_utils


def test_execute_tool_node_returns_completed_with_stream_chunks(monkeypatch):
    """测试目标：工具节点回传 stream 与工具明细；成功标准：tool_calls 被透传。"""

    monkeypatch.setattr(
        agent_utils,
        "invoke_with_failure_policy",
        lambda **_kwargs: (
            "执行成功",
            {
                "stream_chunks": ["执", "行"],
                "tool_call_details": [
                    {
                        "tool_name": "get_order_list",
                        "tool_input": {"page_num": 1},
                        "tool_output": {"list": []},
                        "is_error": False,
                        "error_message": "",
                    }
                ],
            },
        ),
    )

    result = agent_utils.execute_tool_node(
        llm=object(),
        messages=[],
        tools=[],
        enable_stream=True,
        failure_policy={},
        fallback_content="fallback",
        fallback_error="tool failed",
    )
    assert result.status == "completed"
    assert result.error is None
    assert result.content == "执行成功"
    assert result.stream_chunks == ["执", "行"]
    assert result.tool_calls == [
        {
            "tool_name": "get_order_list",
            "tool_input": {"page_num": 1},
            "tool_output": {"list": []},
            "is_error": False,
            "error_message": "",
        }
    ]


def test_execute_tool_node_marks_failed_on_error_marker(monkeypatch):
    monkeypatch.setattr(
        agent_utils,
        "invoke_with_failure_policy",
        lambda **_kwargs: ("__ERROR__: 数据不可信", {}),
    )

    result = agent_utils.execute_tool_node(
        llm=object(),
        messages=[],
        tools=[],
        enable_stream=False,
        failure_policy={},
        fallback_content="fallback",
        fallback_error="tool failed",
    )
    assert result.status == "failed"
    assert result.error == "数据不可信"
    assert result.content == "数据不可信"


def test_execute_tool_node_marks_failed_on_threshold(monkeypatch):
    monkeypatch.setattr(
        agent_utils,
        "invoke_with_failure_policy",
        lambda **_kwargs: (
            "工具错误",
            {"threshold_hit": True, "threshold_reason": "工具失败达到阈值"},
        ),
    )

    result = agent_utils.execute_tool_node(
        llm=object(),
        messages=[],
        tools=[],
        enable_stream=False,
        failure_policy={},
        fallback_content="fallback",
        fallback_error="tool failed",
    )
    assert result.status == "failed"
    assert result.error == "工具失败达到阈值"
    assert result.content == "工具错误"


def test_execute_tool_node_returns_fallback_on_exception(monkeypatch):
    def _raise(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(agent_utils, "invoke_with_failure_policy", _raise)
    result = agent_utils.execute_tool_node(
        llm=object(),
        messages=[],
        tools=[],
        enable_stream=False,
        failure_policy={},
        fallback_content="兜底文案",
        fallback_error="tool failed",
    )
    assert result.status == "failed"
    assert result.content == "兜底文案"
    assert result.error == "tool failed: boom"


def test_execute_text_node_success(monkeypatch):
    monkeypatch.setattr(agent_utils, "invoke", lambda *_args, **_kwargs: "文本结果")
    result = agent_utils.execute_text_node(
        llm=object(),
        messages=[],
        fallback_content="fallback",
        fallback_error="text failed",
    )
    assert result.status == "completed"
    assert result.content == "文本结果"
    assert result.error is None


def test_execute_text_node_fallback_on_exception(monkeypatch):
    def _raise(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(agent_utils, "invoke", _raise)
    result = agent_utils.execute_text_node(
        llm=object(),
        messages=[],
        fallback_content="文本兜底",
        fallback_error="text failed",
    )
    assert result.status == "failed"
    assert result.content == "文本兜底"
    assert result.error == "text failed: boom"


def test_build_standard_node_update_writes_results_and_step_outputs():
    """测试目标：标准节点更新包含 execution_trace；成功标准：trace 字段齐全。"""

    state = {"results": {"chat": {"content": "old"}}}
    runtime = {"step_id": "s1"}
    execution_result = agent_utils.NodeExecutionResult(
        content="订单处理完成",
        status="completed",
        model_name="qwen3-max",
        input_messages=[{"role": "system", "content": "请处理订单"}],
        tool_calls=[],
        stream_chunks=["订", "单"],
    )

    result = agent_utils.build_standard_node_update(
        state=state,
        runtime=runtime,
        node_name="order_agent",
        result_key="order",
        execution_result=execution_result,
        is_end=True,
    )
    assert result["results"]["order"]["content"] == "订单处理完成"
    assert result["results"]["order"]["is_end"] is True
    assert result["results"]["order"]["stream_chunks"] == ["订", "单"]
    assert result["step_outputs"]["s1"]["status"] == "completed"
    assert result["step_outputs"]["s1"]["node_name"] == "order_agent"
    assert result["execution_traces"] == [
        {
            "node_name": "order_agent",
            "model_name": "qwen3-max",
            "input_messages": [{"role": "system", "content": "请处理订单"}],
            "output_text": "订单处理完成",
            "tool_calls": [],
        }
    ]


def test_build_standard_node_update_skips_step_output_without_step_id():
    state = {"results": {}}
    runtime = {"step_id": ""}
    execution_result = agent_utils.NodeExecutionResult(
        content="商品处理完成",
        status="completed",
    )

    result = agent_utils.build_standard_node_update(
        state=state,
        runtime=runtime,
        node_name="product_agent",
        result_key="product",
        execution_result=execution_result,
        is_end=False,
    )
    assert result["results"]["product"]["content"] == "商品处理完成"
    assert result["results"]["product"]["is_end"] is False
    assert "step_outputs" not in result
