import json
from types import SimpleNamespace

from langchain_core.messages import HumanMessage
from langgraph.constants import END

from app.agent.client.domain.consultation import agent as consultation_agent_module
from app.agent.client.domain.consultation import graph as consultation_graph_module
from app.agent.client.domain.consultation import helpers as consultation_helper_module
from app.agent.client.domain.consultation import node as consultation_node_module
from app.agent.client.domain.consultation.nodes import final_diagnosis_node as final_diagnosis_node_module
from app.agent.client.domain.consultation.nodes import question_interrupt_node as question_interrupt_node_module
from app.agent.client.domain.consultation.nodes import question_node as question_node_module
from app.agent.client.domain.consultation.nodes import response_node as response_node_module
from app.agent.client.domain.consultation.nodes import route_node as route_node_module
from app.agent.client.domain.consultation.state import (
    CONSULTATION_STATUS_COLLECTING,
    CONSULTATION_STATUS_COMPLETED,
)
from app.core.config_sync import AgentChatModelSlot


def test_consultation_node_module_reexports_public_symbols():
    assert consultation_node_module.consultation_agent is consultation_agent_module.consultation_agent
    assert consultation_node_module.build_consultation_graph is consultation_graph_module.build_consultation_graph
    assert consultation_node_module._CONSULTATION_GRAPH is consultation_graph_module._CONSULTATION_GRAPH
    assert consultation_node_module.consultation_route_node is route_node_module.consultation_route_node
    assert consultation_node_module.consultation_response_node is response_node_module.consultation_response_node
    assert consultation_node_module.consultation_question_node is question_node_module.consultation_question_node
    assert (
            consultation_node_module.consultation_question_interrupt_node
            is question_interrupt_node_module.consultation_question_interrupt_node
    )
    assert (
            consultation_node_module.consultation_final_diagnosis_node
            is final_diagnosis_node_module.consultation_final_diagnosis_node
    )


def test_consultation_agent_maps_nested_result_to_parent_state(monkeypatch):
    monkeypatch.setattr(
        consultation_agent_module,
        "_CONSULTATION_GRAPH",
        SimpleNamespace(
            invoke=lambda _state, config=None: {
                "consultation_outputs": {
                    "final_diagnosis": {"text": "这是最终诊断结果"},
                },
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


def test_build_llm_agent_uses_route_slot_defaults(monkeypatch):
    captured_llm_kwargs: dict[str, object] = {}
    captured_agent_kwargs: dict[str, object] = {}

    def _fake_create_agent_chat_llm(**kwargs):
        captured_llm_kwargs.update(kwargs)
        return SimpleNamespace(model_name="consultation-route-model")

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
        temperature=0.0,
        slot=AgentChatModelSlot.CLIENT_ROUTE,
    )

    assert agent is not None
    assert llm_model_name == "consultation-route-model"
    assert captured_llm_kwargs["slot"] is AgentChatModelSlot.CLIENT_ROUTE
    assert captured_llm_kwargs["temperature"] == 0.0
    assert captured_llm_kwargs["think"] is False
    assert captured_agent_kwargs["model"].model_name == "consultation-route-model"
    assert "tools" not in captured_agent_kwargs


def test_build_llm_agent_passes_tools_and_extra_middleware(monkeypatch):
    captured_agent_kwargs: dict[str, object] = {}
    tool_marker = object()
    middleware_marker = object()

    monkeypatch.setattr(
        consultation_helper_module,
        "create_agent_chat_llm",
        lambda **_kwargs: SimpleNamespace(model_name="consultation-question-model"),
    )
    monkeypatch.setattr(
        consultation_helper_module,
        "create_agent",
        lambda **kwargs: captured_agent_kwargs.update(kwargs) or object(),
    )

    consultation_helper_module.build_llm_agent(
        state={"task_difficulty": "normal"},
        prompt_text="这是 consultation 测试提示词。",
        temperature=0.2,
        slot=AgentChatModelSlot.CLIENT_CONSULTATION_QUESTION,
        tools=[tool_marker],
        extra_middleware=[middleware_marker],
    )

    assert captured_agent_kwargs["tools"] == [tool_marker]
    assert middleware_marker in captured_agent_kwargs["middleware"]


def test_route_functions_return_expected_next_node():
    assert consultation_graph_module._route_after_consultation_route(
        {"consultation_route": {"next_action": "reply_only"}}
    ) == "response_node"
    assert consultation_graph_module._route_after_consultation_route(
        {"consultation_route": {"next_action": "ask_followup"}}
    ) == "collecting_fanout_node"
    assert consultation_graph_module._route_after_consultation_route(
        {"consultation_route": {"next_action": "final_diagnosis"}}
    ) == "final_diagnosis_node"
    assert consultation_graph_module._route_after_consultation_response(
        {"consultation_route": {"next_action": "reply_only"}}
    ) == END
    assert consultation_graph_module._route_after_consultation_response(
        {"consultation_route": {"next_action": "ask_followup"}}
    ) == "parallel_merge_node"
    assert consultation_graph_module._route_after_parallel_merge({"diagnosis_ready": True}) == (
        "final_diagnosis_node"
    )
    assert consultation_graph_module._route_after_parallel_merge({"diagnosis_ready": False}) == (
        "question_interrupt_node"
    )


def test_consultation_route_node_outputs_structured_route(monkeypatch):
    payload_text = json.dumps(
        {
            "next_action": "reply_only",
            "consultation_mode": "simple_medical",
            "reason": "用户当前是在问缓解建议，不需要继续缩小诊断范围。",
        },
        ensure_ascii=False,
    )

    monkeypatch.setattr(
        route_node_module,
        "build_llm_agent",
        lambda **_kwargs: (object(), "route-model"),
    )
    monkeypatch.setattr(
        route_node_module,
        "agent_invoke",
        lambda _agent, _messages: SimpleNamespace(
            payload={"messages": [SimpleNamespace(content=payload_text)]},
            content=payload_text,
        ),
    )
    monkeypatch.setattr(
        route_node_module,
        "record_agent_trace",
        lambda **_kwargs: {
            "text": payload_text,
            "is_usage_complete": True,
            "usage": None,
            "tool_calls": [],
        },
    )

    result = route_node_module.consultation_route_node(
        {
            "history_messages": [HumanMessage(content="我嗓子疼有什么建议")],
            "execution_traces": [],
            "consultation_progress": {
                "asked_followups": [],
                "asked_slots": [],
                "answered_slots": {},
                "pending_slot_key": "",
            },
        }
    )

    assert result["consultation_route"]["next_action"] == "reply_only"
    assert result["consultation_route"]["consultation_mode"] == "simple_medical"
    assert result["route_trace"]["node_name"] == "consultation_route_node"


def test_consultation_response_node_streams_reply_only_text(monkeypatch):
    answer_deltas: list[str] = []
    thinking_deltas: list[str] = []

    monkeypatch.setattr(
        response_node_module,
        "build_llm_agent",
        lambda **_kwargs: (object(), "response-model"),
    )

    def _fake_agent_stream(_agent, _messages, on_model_delta=None, on_thinking_delta=None):
        assert on_model_delta is not None
        assert on_thinking_delta is not None
        on_thinking_delta("先分析缓解动作")
        on_model_delta("先多喝温水，今天先别吃辛辣刺激。")
        return {
            "streamed_text": "先多喝温水，今天先别吃辛辣刺激。",
            "streamed_thinking": "先分析缓解动作",
            "latest_state": {"messages": []},
        }

    monkeypatch.setattr(response_node_module, "agent_stream", _fake_agent_stream)
    monkeypatch.setattr(response_node_module, "emit_answer_delta", answer_deltas.append)
    monkeypatch.setattr(response_node_module, "emit_thinking_delta", thinking_deltas.append)
    monkeypatch.setattr(
        response_node_module,
        "record_agent_trace",
        lambda **_kwargs: {
            "text": "先多喝温水，今天先别吃辛辣刺激。",
            "is_usage_complete": True,
            "usage": None,
            "tool_calls": [],
        },
    )

    result = response_node_module.consultation_response_node(
        {
            "history_messages": [HumanMessage(content="喉咙痛现在怎么办")],
            "execution_traces": [],
            "consultation_route": {
                "next_action": "reply_only",
                "consultation_mode": "simple_medical",
                "reason": "不需要继续追问。",
            },
            "consultation_progress": {
                "asked_followups": [],
                "asked_slots": [],
                "answered_slots": {},
                "pending_slot_key": "",
            },
        }
    )

    assert answer_deltas == ["先多喝温水，今天先别吃辛辣刺激。"]
    assert thinking_deltas == ["先分析缓解动作"]
    assert result["consultation_status"] == CONSULTATION_STATUS_COMPLETED
    assert result["consultation_outputs"]["response"]["text"] == "先多喝温水，今天先别吃辛辣刺激。"
    assert result["result"] == "先多喝温水，今天先别吃辛辣刺激。"


def test_consultation_question_node_requests_followup_with_slot_key(monkeypatch):
    captured_build_llm_kwargs: dict[str, object] = {}
    payload_text = json.dumps(
        {
            "diagnosis_ready": False,
            "question_reply_text": "你提到低烧和流鼻涕，这更像上呼吸道轻症方向，但还要确认咳嗽有没有痰。",
            "question_text": "咳嗽有痰吗？",
            "options": ["没有痰", "白痰", "黄痰", "不确定"],
            "slot_key": "cough_sputum",
        },
        ensure_ascii=False,
    )

    monkeypatch.setattr(
        question_node_module,
        "build_llm_agent",
        lambda **kwargs: captured_build_llm_kwargs.update(kwargs) or (object(), "question-model"),
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
            "consultation_progress": {
                "asked_followups": [],
                "asked_slots": [],
                "answered_slots": {},
                "pending_slot_key": "",
            },
        }
    )

    assert result["consultation_status"] == CONSULTATION_STATUS_COLLECTING
    assert result["diagnosis_ready"] is False
    assert "上呼吸道轻症方向" in result["consultation_outputs"]["question"]["reply_text"]
    assert result["consultation_outputs"]["question"]["question_text"] == "咳嗽有痰吗？"
    assert result["consultation_outputs"]["question"]["options"] == ["没有痰", "白痰", "黄痰", "不确定"]
    assert result["consultation_progress"]["pending_slot_key"] == "cough_sputum"
    assert [tool.name for tool in captured_build_llm_kwargs["tools"]] == [
        "search_symptom_candidates",
        "query_disease_candidates_by_symptoms",
        "query_followup_symptom_candidates",
    ]


def test_consultation_question_node_avoids_duplicate_followup(monkeypatch):
    payload_text = json.dumps(
        {
            "diagnosis_ready": False,
            "question_reply_text": "还需要确认体温。",
            "question_text": "现在体温多少？",
            "options": ["未测量", "37.3以下", "37.3到38", "38以上"],
            "slot_key": "temperature",
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
            "history_messages": [HumanMessage(content="我已经说过低烧了")],
            "consultation_progress": {
                "asked_followups": [
                    {
                        "slot_key": "temperature",
                        "question_text": "现在体温多少？",
                        "options": ["未测量", "37.3以下", "37.3到38", "38以上"],
                        "answer_text": "低烧",
                    }
                ],
                "asked_slots": ["temperature"],
                "answered_slots": {"temperature": "低烧"},
                "pending_slot_key": "",
            },
        }
    )

    assert result["consultation_status"] == CONSULTATION_STATUS_COMPLETED
    assert result["diagnosis_ready"] is True
    assert result["question_trace"]["node_context"]["duplicate_followup"] is True


def test_consultation_parallel_merge_node_combines_outputs_and_traces():
    result = consultation_graph_module.consultation_parallel_merge_node(
        {
            "diagnosis_ready": False,
            "consultation_outputs": {
                "response": {"text": "先按感冒轻症方向处理。"},
                "question": {"reply_text": "但还要确认咳嗽有没有痰。"},
            },
            "execution_traces": [
                {
                    "sequence": 1,
                    "node_name": "consultation_route_node",
                    "model_name": "route-model",
                    "status": "success",
                    "output_text": "ask_followup",
                    "llm_usage_complete": True,
                    "llm_token_usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    "tool_calls": [],
                    "node_context": None,
                }
            ],
            "response_trace": {
                "sequence": 0,
                "node_name": "consultation_response_node",
                "model_name": "response-model",
                "status": "success",
                "output_text": "先按感冒轻症方向处理。",
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
                "output_text": "但还要确认咳嗽有没有痰。",
                "llm_usage_complete": True,
                "llm_token_usage": {"prompt_tokens": 4, "completion_tokens": 4, "total_tokens": 8},
                "tool_calls": [],
                "node_context": None,
            },
        }
    )

    assert result["consultation_outputs"]["question"]["ai_reply_text"] == (
        "先按感冒轻症方向处理。\n\n但还要确认咳嗽有没有痰。"
    )
    assert [trace["node_name"] for trace in result["execution_traces"]] == [
        "consultation_route_node",
        "consultation_response_node",
        "consultation_question_node",
    ]
    assert result["token_usage"]["total_tokens"] == 15


def test_consultation_question_interrupt_node_appends_progress_and_history(monkeypatch):
    monkeypatch.setattr(
        question_interrupt_node_module,
        "interrupt",
        lambda _payload: "低烧",
    )

    result = question_interrupt_node_module.consultation_question_interrupt_node(
        {
            "history_messages": [HumanMessage(content="我咳嗽流鼻涕")],
            "consultation_outputs": {
                "response": {"text": "先按上呼吸道轻症方向处理。"},
                "question": {
                    "reply_text": "目前更偏向常见轻症方向，但还要确认有没有发热。",
                    "question_text": "现在有发热吗？",
                    "options": ["没有发热", "低烧", "高烧", "不确定"],
                    "ai_reply_text": "先按上呼吸道轻症方向处理。\n\n目前更偏向常见轻症方向，但还要确认有没有发热。",
                },
            },
            "consultation_progress": {
                "asked_followups": [],
                "asked_slots": [],
                "answered_slots": {},
                "pending_slot_key": "temperature",
            },
            "execution_traces": [],
        }
    )

    assert result["history_messages"][-2].content == (
        "先按上呼吸道轻症方向处理。\n\n目前更偏向常见轻症方向，但还要确认有没有发热。"
    )
    assert result["history_messages"][-1].content == "低烧"
    assert result["consultation_progress"]["asked_slots"] == ["temperature"]
    assert result["consultation_progress"]["answered_slots"] == {"temperature": "低烧"}
    assert result["consultation_outputs"]["interrupt"]["payload"]["question_text"] == "现在有发热吗？"
    assert result["last_resume_text"] == "低烧"


def test_consultation_final_diagnosis_node_uses_tool_agent_and_thinking(monkeypatch):
    answer_deltas: list[str] = []
    thinking_deltas: list[str] = []
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
        assert on_thinking_delta is not None
        on_thinking_delta("先看工具搜索结果")
        on_model_delta("结合你的情况，更像是常见感冒方向。")
        return {
            "streamed_text": "结合你的情况，更像是常见感冒方向。",
            "streamed_thinking": "先看工具搜索结果",
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
    monkeypatch.setattr(final_diagnosis_node_module, "emit_thinking_delta", thinking_deltas.append)
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
            "execution_traces": [],
            "consultation_progress": {
                "asked_followups": [],
                "asked_slots": [],
                "answered_slots": {},
                "pending_slot_key": "",
            },
        }
    )

    tool_names = [tool.name for tool in captured_agent_kwargs["tools"]]

    assert captured_llm_kwargs["slot"] is AgentChatModelSlot.CLIENT_CONSULTATION_FINAL_DIAGNOSIS
    assert captured_llm_kwargs["temperature"] == 0.2
    assert captured_llm_kwargs["think"] is False
    assert tool_names == [
        "search_symptom_candidates",
        "query_disease_candidates_by_symptoms",
        "query_disease_detail",
        "query_followup_symptom_candidates",
        "search_products",
        "get_product_detail",
        "get_product_spec",
        "send_product_purchase_card",
    ]
    assert answer_deltas == ["结合你的情况，更像是常见感冒方向。"]
    assert thinking_deltas == ["先看工具搜索结果"]
    assert result["consultation_status"] == CONSULTATION_STATUS_COMPLETED
    assert result["diagnosis_ready"] is True
    assert result["consultation_outputs"]["final_diagnosis"]["text"] == "结合你的情况，更像是常见感冒方向。"
    assert result["result"] == "结合你的情况，更像是常见感冒方向。"
