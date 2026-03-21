import json
from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from app.agent.client.domain.consultation import agent as consultation_agent_module
from app.agent.client.domain.consultation import graph as consultation_graph_module
from app.agent.client.domain.consultation import helpers as consultation_helper_module
from app.agent.client.domain.consultation import node as consultation_node_module
from app.agent.client.domain.consultation.nodes import comfort_node as comfort_node_module
from app.agent.client.domain.consultation.nodes import final_diagnosis_node as final_diagnosis_node_module
from app.agent.client.domain.consultation.nodes import question_interrupt_node as question_interrupt_node_module
from app.agent.client.domain.consultation.nodes import question_node as question_node_module
from app.agent.client.domain.consultation.state import (
    CONSULTATION_STATUS_COLLECTING,
    CONSULTATION_STATUS_COMPLETED,
)
from app.core.config_sync import AgentChatModelSlot


def test_consultation_node_module_reexports_public_symbols():
    assert consultation_node_module.consultation_agent is consultation_agent_module.consultation_agent
    assert consultation_node_module.build_consultation_graph is consultation_graph_module.build_consultation_graph
    assert consultation_node_module._CONSULTATION_GRAPH is consultation_graph_module._CONSULTATION_GRAPH
    assert consultation_node_module.consultation_comfort_node is comfort_node_module.consultation_comfort_node
    assert consultation_node_module.consultation_question_node is question_node_module.consultation_question_node
    assert (
            consultation_node_module.consultation_question_interrupt_node
            is question_interrupt_node_module.consultation_question_interrupt_node
    )
    assert (
            consultation_node_module.consultation_final_diagnosis_node
            is final_diagnosis_node_module.consultation_final_diagnosis_node
    )


def test_consultation_agent_maps_subgraph_result_to_parent_state(monkeypatch):
    monkeypatch.setattr(
        consultation_agent_module,
        "_CONSULTATION_GRAPH",
        SimpleNamespace(
            invoke=lambda _state, config=None: {
                "final_text": "这是最终诊断结果",
                "execution_traces": [
                    {
                        "sequence": 0,
                        "node_name": "consultation_final_diagnosis_node",
                        "model_name": "diagnosis-model",
                        "status": "success",
                        "output_text": "这是最终诊断结果",
                        "llm_usage_complete": True,
                        "llm_token_usage": {
                            "prompt_tokens": 2,
                            "completion_tokens": 3,
                            "total_tokens": 5,
                        },
                        "tool_calls": [],
                        "node_context": None,
                    }
                ],
                "__interrupt__": None,
            }
        ),
    )
    monkeypatch.setattr(
        consultation_agent_module,
        "build_consultation_graph_config",
        lambda config: config,
    )

    result = consultation_agent_module.consultation_agent(
        {
            "history_messages": [HumanMessage(content="我是不是感冒了")],
            "execution_traces": [
                {
                    "sequence": 1,
                    "node_name": "gateway_router",
                    "model_name": "gateway-model",
                    "status": "success",
                    "output_text": "consultation_agent",
                    "llm_usage_complete": True,
                    "llm_token_usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                    "tool_calls": [],
                    "node_context": None,
                }
            ],
            "routing": {"task_difficulty": "normal"},
        },
        config={"configurable": {"thread_id": "conv-1"}},
    )

    assert result["result"] == "这是最终诊断结果"
    assert result["messages"][0].content == "这是最终诊断结果"
    assert [trace["node_name"] for trace in result["execution_traces"]] == [
        "gateway_router",
        "consultation_final_diagnosis_node",
    ]
    assert [trace["sequence"] for trace in result["execution_traces"]] == [1, 2]
    assert result["token_usage"]["total_tokens"] == 7


def test_build_llm_agent_uses_consultation_slot(monkeypatch):
    captured_llm_kwargs: dict[str, object] = {}
    captured_agent_kwargs: dict[str, object] = {}

    def _fake_create_agent_chat_llm(**kwargs):
        captured_llm_kwargs.update(kwargs)
        return SimpleNamespace(model_name="consultation-complex-model")

    def _fake_create_agent(**kwargs):
        captured_agent_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(
        consultation_helper_module,
        "create_agent_chat_llm",
        _fake_create_agent_chat_llm,
    )
    monkeypatch.setattr(
        consultation_helper_module,
        "create_agent",
        _fake_create_agent,
    )

    agent, llm_model_name = consultation_helper_module.build_llm_agent(
        state={"task_difficulty": "normal"},
        prompt_text="这是 consultation 测试提示词。",
        temperature=0.35,
        slot=AgentChatModelSlot.CLIENT_CONSULTATION_QUESTION,
    )

    assert agent is not None
    assert llm_model_name == "consultation-complex-model"
    assert captured_llm_kwargs["slot"] is AgentChatModelSlot.CLIENT_CONSULTATION_QUESTION
    assert captured_llm_kwargs["temperature"] == 0.35
    assert captured_llm_kwargs["think"] is False
    assert captured_agent_kwargs["model"].model_name == "consultation-complex-model"


def test_route_functions_return_expected_next_node():
    assert consultation_graph_module._route_from_entry({"diagnosis_ready": True}) == (
        "consultation_final_diagnosis_node"
    )
    assert consultation_graph_module._route_from_entry({"diagnosis_ready": False}) == (
        "consultation_collecting_fanout_node"
    )
    assert consultation_graph_module._route_after_parallel_merge({"diagnosis_ready": True}) == (
        "consultation_final_diagnosis_node"
    )
    assert consultation_graph_module._route_after_parallel_merge({"diagnosis_ready": False}) == (
        "consultation_question_interrupt_node"
    )


def test_consultation_comfort_node_streams_answer_and_builds_trace(monkeypatch):
    emitted_chunks: list[str] = []

    monkeypatch.setattr(
        comfort_node_module,
        "build_llm_agent",
        lambda **_kwargs: (object(), "comfort-model"),
    )

    def _fake_agent_stream(_agent, _messages, on_model_delta=None, on_thinking_delta=None):
        assert on_thinking_delta is None
        if on_model_delta is not None:
            on_model_delta("先别太担心，")
            on_model_delta("我们先按现在的情况判断。")
        return {
            "streamed_text": "先别太担心，我们先按现在的情况判断。",
            "latest_state": {"messages": []},
        }

    monkeypatch.setattr(comfort_node_module, "agent_stream", _fake_agent_stream)
    monkeypatch.setattr(comfort_node_module, "emit_answer_delta", emitted_chunks.append)
    monkeypatch.setattr(
        comfort_node_module,
        "record_agent_trace",
        lambda **_kwargs: {
            "text": "先别太担心，我们先按现在的情况判断。",
            "is_usage_complete": True,
            "usage": None,
            "tool_calls": [],
        },
    )

    result = comfort_node_module.consultation_comfort_node(
        {
            "history_messages": [HumanMessage(content="我咳嗽流鼻涕")],
            "task_difficulty": "normal",
        }
    )

    assert emitted_chunks == ["先别太担心，", "我们先按现在的情况判断。"]
    assert result["comfort_text"] == "先别太担心，我们先按现在的情况判断。"
    assert result["comfort_trace"]["node_name"] == "consultation_comfort_node"


def test_consultation_question_node_requests_more_info(monkeypatch):
    payload_text = json.dumps(
        {
            "diagnosis_ready": False,
            "question_reply_text": "你提到低烧和流鼻涕，这更像上呼吸道轻症方向，但还要确认咳嗽有没有痰，才能更好区分是刺激性咳嗽还是已有分泌物。",
            "question_text": "咳嗽有痰吗？",
            "options": ["没有痰", "白痰", "黄痰", "不确定"],
        },
        ensure_ascii=False,
    )

    monkeypatch.setattr(
        question_node_module,
        "build_llm_agent",
        lambda **_kwargs: (object(), "question-model"),
    )
    monkeypatch.setattr(
        question_node_module,
        "agent_invoke",
        lambda _agent, _messages: SimpleNamespace(
            payload={"messages": [SimpleNamespace(content=payload_text)]},
            content=payload_text,
        ),
    )
    monkeypatch.setattr(
        question_node_module,
        "record_agent_trace",
        lambda **_kwargs: {
            "text": payload_text,
            "is_usage_complete": True,
            "usage": None,
            "tool_calls": [],
        },
    )

    result = question_node_module.consultation_question_node(
        {
            "history_messages": [HumanMessage(content="我低烧两天，还流鼻涕")],
            "task_difficulty": "normal",
        }
    )

    assert result["consultation_status"] == CONSULTATION_STATUS_COLLECTING
    assert result["diagnosis_ready"] is False
    assert "上呼吸道轻症方向" in result["question_reply_text"]
    assert result["pending_question_text"] == "咳嗽有痰吗？"
    assert result["pending_question_options"] == ["没有痰", "白痰", "黄痰", "不确定"]
    assert result["question_trace"]["node_name"] == "consultation_question_node"


def test_consultation_question_node_marks_diagnosis_ready(monkeypatch):
    payload_text = json.dumps(
        {
            "diagnosis_ready": True,
            "question_reply_text": None,
            "question_text": None,
            "options": [],
        },
        ensure_ascii=False,
    )

    monkeypatch.setattr(
        question_node_module,
        "build_llm_agent",
        lambda **_kwargs: (object(), "question-model"),
    )
    monkeypatch.setattr(
        question_node_module,
        "agent_invoke",
        lambda _agent, _messages: SimpleNamespace(
            payload={"messages": [SimpleNamespace(content=payload_text)]},
            content=payload_text,
        ),
    )
    monkeypatch.setattr(
        question_node_module,
        "record_agent_trace",
        lambda **_kwargs: {
            "text": payload_text,
            "is_usage_complete": True,
            "usage": None,
            "tool_calls": [],
        },
    )

    result = question_node_module.consultation_question_node(
        {
            "history_messages": [HumanMessage(content="我低烧两天，黄痰，鼻塞明显")],
            "task_difficulty": "normal",
        }
    )

    assert result["consultation_status"] == CONSULTATION_STATUS_COMPLETED
    assert result["diagnosis_ready"] is True
    assert result["question_reply_text"] == ""
    assert result["pending_question_text"] == ""
    assert result["pending_question_options"] == []


def test_consultation_parallel_merge_node_combines_reply_and_traces():
    result = consultation_graph_module.consultation_parallel_merge_node(
        {
            "diagnosis_ready": False,
            "comfort_text": "先别太担心，我们先按现在的情况判断。",
            "question_reply_text": "你提到低烧和流鼻涕，这让我们更偏向常见上呼吸道轻症方向。",
            "execution_traces": [
                {
                    "sequence": 1,
                    "node_name": "gateway_router",
                    "model_name": "gateway-model",
                    "status": "success",
                    "output_text": "consultation_agent",
                    "llm_usage_complete": True,
                    "llm_token_usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    "tool_calls": [],
                    "node_context": None,
                }
            ],
            "comfort_trace": {
                "sequence": 0,
                "node_name": "consultation_comfort_node",
                "model_name": "comfort-model",
                "status": "success",
                "output_text": "先别太担心，我们先按现在的情况判断。",
                "llm_usage_complete": True,
                "llm_token_usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
                "tool_calls": [],
                "node_context": None,
            },
            "question_trace": {
                "sequence": 0,
                "node_name": "consultation_question_node",
                "model_name": "question-model",
                "status": "success",
                "output_text": "你提到低烧和流鼻涕，这让我们更偏向常见上呼吸道轻症方向。",
                "llm_usage_complete": True,
                "llm_token_usage": {"prompt_tokens": 4, "completion_tokens": 4, "total_tokens": 8},
                "tool_calls": [],
                "node_context": None,
            },
        }
    )

    assert result["pending_ai_reply_text"] == (
        "先别太担心，我们先按现在的情况判断。\n\n"
        "你提到低烧和流鼻涕，这让我们更偏向常见上呼吸道轻症方向。"
    )
    assert [trace["node_name"] for trace in result["execution_traces"]] == [
        "gateway_router",
        "consultation_comfort_node",
        "consultation_question_node",
    ]
    assert [trace["sequence"] for trace in result["execution_traces"]] == [1, 2, 3]
    assert result["token_usage"]["total_tokens"] == 15


def test_consultation_question_interrupt_node_appends_combined_reply_on_resume(monkeypatch):
    monkeypatch.setattr(
        question_interrupt_node_module,
        "interrupt",
        lambda _payload: "低烧",
    )

    result = question_interrupt_node_module.consultation_question_interrupt_node(
        {
            "history_messages": [HumanMessage(content="我咳嗽流鼻涕")],
            "comfort_text": "先别太担心，我们先按现在的情况判断。",
            "question_reply_text": "目前更偏向常见轻症方向，但还要确认有没有发热。",
            "pending_ai_reply_text": "先别太担心，我们先按现在的情况判断。\n\n目前更偏向常见轻症方向，但还要确认有没有发热。",
            "pending_question_text": "现在有发热吗？",
            "pending_question_options": ["没有发热", "低烧", "高烧", "不确定"],
            "execution_traces": [],
        }
    )

    assert result["consultation_status"] == CONSULTATION_STATUS_COLLECTING
    assert result["history_messages"][-2].content == (
        "先别太担心，我们先按现在的情况判断。\n\n目前更偏向常见轻症方向，但还要确认有没有发热。"
    )
    assert result["history_messages"][-1].content == "低烧"
    assert result["interrupt_payload"]["reply_text"] == "目前更偏向常见轻症方向，但还要确认有没有发热。"
    assert result["interrupt_payload"]["question_text"] == "现在有发热吗？"
    assert result["last_resume_text"] == "低烧"
    assert result["interrupt_trace"]["node_name"] == "consultation_question_interrupt_node"


def test_split_consultation_stream_text_preserves_original_text_order():
    source_text = (
        "结合你目前的描述，更像是常见上呼吸道不适。"
        "如果出现持续高热、呼吸困难或胸痛，请尽快线下就医。"
        "\n\n"
        "可考虑的药品有：感冒灵颗粒、板蓝根颗粒。"
    )

    chunks = consultation_helper_module.split_consultation_stream_text(
        source_text,
        max_chunk_length=18,
    )

    assert "".join(chunks) == source_text
    assert any(chunk.endswith("。") for chunk in chunks)
    assert any("\n\n" in chunk for chunk in chunks)
    assert all(chunk for chunk in chunks)


def test_consultation_final_diagnosis_node_uses_complex_slot_tool_agent(monkeypatch):
    answer_deltas: list[str] = []
    captured_agent_kwargs: dict[str, object] = {}
    captured_llm_kwargs: dict[str, object] = {}

    def _fake_create_agent_chat_llm(**kwargs):
        captured_llm_kwargs.update(kwargs)
        return SimpleNamespace(model_name="complex-model")

    def _fake_create_agent(**kwargs):
        captured_agent_kwargs.update(kwargs)
        return object()

    def _fake_agent_stream(_agent, _messages, on_model_delta=None, on_thinking_delta=None):
        assert on_model_delta is not None
        assert on_thinking_delta is None
        on_model_delta("结合你的情况，更像是常见感冒方向。")
        return {
            "streamed_text": "结合你的情况，更像是常见感冒方向。",
            "streamed_thinking": "",
            "latest_state": {"messages": []},
        }

    monkeypatch.setattr(
        final_diagnosis_node_module,
        "create_agent_chat_llm",
        _fake_create_agent_chat_llm,
    )
    monkeypatch.setattr(final_diagnosis_node_module, "create_agent", _fake_create_agent)
    monkeypatch.setattr(final_diagnosis_node_module, "agent_stream", _fake_agent_stream)
    monkeypatch.setattr(final_diagnosis_node_module, "emit_answer_delta", answer_deltas.append)
    monkeypatch.setattr(
        final_diagnosis_node_module,
        "record_agent_trace",
        lambda **_kwargs: {
            "text": "结合你的情况，更像是常见感冒方向。",
            "model_name": "trace-diagnosis-model",
            "is_usage_complete": True,
            "usage": None,
            "tool_calls": [
                {
                    "tool_name": "search_products",
                    "tool_call_id": "tool-call-1",
                    "tool_input": {"keyword": "感冒药"},
                },
                {
                    "tool_name": "send_product_purchase_card",
                    "tool_call_id": "tool-call-2",
                    "tool_input": {"items": [{"productId": 101, "quantity": 1}]},
                },
            ],
        },
    )

    result = final_diagnosis_node_module.consultation_final_diagnosis_node(
        {
            "history_messages": [HumanMessage(content="我咳嗽两天，还有点流鼻涕")],
            "task_difficulty": "high",
            "execution_traces": [],
        }
    )

    tool_names = [tool.name for tool in captured_agent_kwargs["tools"]]

    assert captured_llm_kwargs["slot"] is AgentChatModelSlot.CLIENT_CONSULTATION_FINAL_DIAGNOSIS
    assert captured_llm_kwargs["temperature"] == 0.2
    assert captured_llm_kwargs["think"] is False
    assert tool_names == [
        "search_products",
        "get_product_detail",
        "get_product_spec",
        "send_product_purchase_card",
    ]
    assert any(
        middleware.__class__.__name__ == "ToolCallLimitMiddleware"
        for middleware in captured_agent_kwargs["middleware"]
    )
    assert answer_deltas == ["结合你的情况，更像是常见感冒方向。"]
    assert result["consultation_status"] == CONSULTATION_STATUS_COMPLETED
    assert result["diagnosis_ready"] is True
    assert result["final_text"] == "结合你的情况，更像是常见感冒方向。"
    assert result["diagnosis_trace"]["tool_calls"][0]["tool_name"] == "search_products"
    assert result["diagnosis_trace"]["tool_calls"][1]["tool_name"] == "send_product_purchase_card"
    assert result["diagnosis_trace"]["model_name"] == "complex-model"
    assert result["diagnosis_trace"]["node_context"] is None


def test_consultation_final_diagnosis_node_returns_text_without_purchase_card(monkeypatch):
    answer_deltas: list[str] = []

    monkeypatch.setattr(
        final_diagnosis_node_module,
        "create_agent_chat_llm",
        lambda **_kwargs: SimpleNamespace(model_name="complex-model"),
    )
    monkeypatch.setattr(final_diagnosis_node_module, "create_agent", lambda **_kwargs: object())

    def _fake_agent_stream(_agent, _messages, on_model_delta=None, on_thinking_delta=None):
        assert on_model_delta is not None
        assert on_thinking_delta is None
        on_model_delta("更像是轻度咽喉不适，建议先观察。")
        return {
            "streamed_text": "更像是轻度咽喉不适，建议先观察。",
            "streamed_thinking": "",
            "latest_state": {"messages": []},
        }

    monkeypatch.setattr(final_diagnosis_node_module, "agent_stream", _fake_agent_stream)
    monkeypatch.setattr(final_diagnosis_node_module, "emit_answer_delta", answer_deltas.append)
    monkeypatch.setattr(
        final_diagnosis_node_module,
        "record_agent_trace",
        lambda **_kwargs: {
            "text": "更像是轻度咽喉不适，建议先观察。",
            "model_name": "trace-diagnosis-model",
            "is_usage_complete": True,
            "usage": None,
            "tool_calls": [
                {
                    "tool_name": "search_products",
                    "tool_call_id": "tool-call-1",
                    "tool_input": {"keyword": "咽喉药"},
                }
            ],
        },
    )

    result = final_diagnosis_node_module.consultation_final_diagnosis_node(
        {
            "history_messages": [HumanMessage(content="我喉咙有点不舒服")],
            "task_difficulty": "normal",
            "execution_traces": [],
        }
    )

    assert answer_deltas == ["更像是轻度咽喉不适，建议先观察。"]
    assert result["consultation_status"] == CONSULTATION_STATUS_COMPLETED
    assert result["diagnosis_ready"] is True
    assert result["final_text"] == "更像是轻度咽喉不适，建议先观察。"
    assert [tool_call["tool_name"] for tool_call in result["diagnosis_trace"]["tool_calls"]] == [
        "search_products"
    ]
    assert result["diagnosis_trace"]["node_context"] is None
