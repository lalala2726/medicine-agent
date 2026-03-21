from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from app.agent.client.domain.consultation import agent as consultation_agent_module
from app.agent.client.domain.consultation import graph as consultation_graph_module
from app.agent.client.domain.consultation import helpers as consultation_helper_module
from app.agent.client.domain.consultation import node as consultation_node_module
from app.agent.client.domain.consultation.nodes import final_diagnosis_node as final_diagnosis_node_module
from app.agent.client.domain.consultation.nodes import question_node as question_node_module
from app.agent.client.domain.consultation.nodes import status_node as status_node_module
from app.agent.client.domain.consultation.nodes import stream_response_node as stream_response_node_module
from app.agent.client.domain.consultation.state import (
    CONSULTATION_STATUS_COLLECTING,
    CONSULTATION_STATUS_COMPLETED,
)


def test_consultation_node_module_reexports_public_symbols():
    assert consultation_node_module.consultation_agent is consultation_agent_module.consultation_agent
    assert consultation_node_module.build_consultation_graph is consultation_graph_module.build_consultation_graph
    assert consultation_node_module._CONSULTATION_GRAPH is consultation_graph_module._CONSULTATION_GRAPH
    assert consultation_node_module.consultation_status_node is status_node_module.consultation_status_node
    assert consultation_node_module.consultation_question_card_node is question_node_module.consultation_question_card_node
    assert consultation_node_module.consultation_final_diagnosis_node is (
        final_diagnosis_node_module.consultation_final_diagnosis_node
    )
    assert consultation_node_module.consultation_stream_response_node is (
        consultation_graph_module.consultation_stream_response_node
    )


def test_consultation_agent_maps_subgraph_result_to_parent_state(monkeypatch):
    monkeypatch.setattr(
        consultation_agent_module,
        "_CONSULTATION_GRAPH",
        SimpleNamespace(
            invoke=lambda _state: {
                "final_text": "这是初步判断结果",
                "status_trace": {
                    "sequence": 0,
                    "node_name": "consultation_status_node",
                    "model_name": "status-model",
                    "status": "success",
                    "output_text": "collecting",
                    "llm_usage_complete": True,
                    "llm_token_usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                    "tool_calls": [],
                    "node_context": None,
                },
                "diagnosis_trace": {
                    "sequence": 0,
                    "node_name": "consultation_final_diagnosis_node",
                    "model_name": "diagnosis-model",
                    "status": "success",
                    "output_text": "这是初步判断结果",
                    "llm_usage_complete": True,
                    "llm_token_usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": 3,
                        "total_tokens": 5,
                    },
                    "tool_calls": [],
                    "node_context": None,
                },
            }
        ),
    )

    result = consultation_agent_module.consultation_agent(
        {
            "history_messages": [HumanMessage(content="我是不是感冒了")],
            "execution_traces": [],
            "routing": {"task_difficulty": "normal"},
        }
    )

    assert result["result"] == "这是初步判断结果"
    assert result["messages"][0].content == "这是初步判断结果"
    assert [trace["node_name"] for trace in result["execution_traces"]] == [
        "consultation_status_node",
        "consultation_final_diagnosis_node",
    ]
    assert [trace["sequence"] for trace in result["execution_traces"]] == [1, 2]
    assert result["token_usage"]["total_tokens"] == 7


def test_consultation_status_node_marks_completed_when_json_requests_final(monkeypatch):
    payload_text = (
        '{"should_enter_diagnosis": true, '
        '"consultation_status": "completed"}'
    )

    monkeypatch.setattr(
        status_node_module,
        "build_llm_agent",
        lambda **_kwargs: (object(), "status-model"),
    )
    monkeypatch.setattr(
        status_node_module,
        "agent_invoke",
        lambda _agent, _messages: SimpleNamespace(
            payload={
                "messages": [
                    SimpleNamespace(content=payload_text)
                ]
            },
            content=payload_text,
        ),
    )
    monkeypatch.setattr(
        status_node_module,
        "record_agent_trace",
        lambda **_kwargs: {
            "text": '{"should_enter_diagnosis": true, "consultation_status": "completed"}',
            "is_usage_complete": True,
            "usage": None,
        },
    )

    result = status_node_module.consultation_status_node(
        {
            "history_messages": [HumanMessage(content="我已经咳嗽三天还有低烧")],
            "task_difficulty": "normal",
        }
    )

    assert result["should_enter_diagnosis"] is True
    assert result["consultation_status"] == CONSULTATION_STATUS_COMPLETED
    assert result["status_trace"]["node_name"] == "consultation_status_node"


def test_collecting_response_node_merges_comfort_before_question():
    result = consultation_graph_module.consultation_collecting_response_node(
        {
            "comfort_text": "先别太担心，这更像是常见上呼吸道不适。",
            "question_text": "我还想确认一下，你有没有发热？",
        }
    )

    assert result["consultation_status"] == CONSULTATION_STATUS_COLLECTING
    assert result["final_text"] == (
        "先别太担心，这更像是常见上呼吸道不适。\n\n我还想确认一下，你有没有发热？"
    )


def test_split_consultation_stream_text_preserves_original_text_order():
    source_text = (
        "先别太担心，这更像是常见上呼吸道不适。"
        "如果出现持续高热、呼吸困难或胸痛，请尽快线下就医。"
        "\n\n"
        "我还想确认一下，你有没有发热，以及症状持续了几天？"
    )

    chunks = consultation_helper_module.split_consultation_stream_text(
        source_text,
        max_chunk_length=18,
    )

    assert "".join(chunks) == source_text
    assert any(chunk.endswith("。") for chunk in chunks)
    assert any("\n\n" in chunk for chunk in chunks)
    assert all(chunk for chunk in chunks)


def test_consultation_stream_response_node_emits_answer_deltas_without_rewriting(monkeypatch):
    emitted_texts: list[str] = []

    monkeypatch.setattr(
        stream_response_node_module,
        "emit_consultation_answer_deltas",
        lambda text: emitted_texts.extend(
            consultation_helper_module.split_consultation_stream_text(text)
        ),
    )

    result = stream_response_node_module.consultation_stream_response_node(
        {
            "final_text": "先别太担心。\n\n我还想确认一下，你有没有发热？",
        }
    )

    assert result["final_text"] == "先别太担心。\n\n我还想确认一下，你有没有发热？"
    assert "".join(emitted_texts) == result["final_text"]


def test_consultation_question_card_node_sends_selection_card_when_collecting(monkeypatch):
    captured_calls: list[tuple[str, dict]] = []

    monkeypatch.setattr(
        question_node_module,
        "build_llm_agent",
        lambda **_kwargs: (object(), "question-model"),
    )
    monkeypatch.setattr(
        question_node_module,
        "agent_invoke",
        lambda _agent, _messages: SimpleNamespace(
            payload={
                "messages": [
                    SimpleNamespace(
                        content=(
                            '{"should_enter_diagnosis": false, '
                            '"consultation_status": "collecting", '
                            '"question_text": "为了更准确判断，你现在有没有发热？", '
                            '"options": ["没有发热", "低烧", "高烧", "不确定"]}'
                        )
                    )
                ]
            },
            content="",
        ),
    )
    monkeypatch.setattr(
        question_node_module,
        "record_agent_trace",
        lambda **_kwargs: {
            "text": "为了更准确判断，你现在有没有发热？",
            "is_usage_complete": True,
            "usage": None,
        },
    )
    monkeypatch.setattr(
        question_node_module,
        "invoke_runnable",
        lambda tool_object, payload: (
            captured_calls.append((tool_object.name, payload)),
            "__SUCCESS__",
        )[1],
    )
    monkeypatch.setattr(
        question_node_module,
        "agent_stream",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not use agent_stream")),
        raising=False,
    )

    result = question_node_module.consultation_question_card_node(
        {
            "history_messages": [HumanMessage(content="我咳嗽流鼻涕")],
            "task_difficulty": "normal",
        }
    )

    assert result["consultation_status"] == CONSULTATION_STATUS_COLLECTING
    assert result["should_enter_diagnosis"] is False
    assert result["question_text"] == "为了更准确判断，你现在有没有发热？"
    assert captured_calls == [
        (
            "send_selection_card",
            {
                "title": "为了更准确判断，你现在有没有发热？",
                "options": ["没有发热", "低烧", "高烧", "不确定"],
            },
        )
    ]
    assert result["question_trace"]["tool_calls"][0]["tool_name"] == "send_selection_card"


def test_consultation_final_diagnosis_node_sends_purchase_card_with_default_quantity(monkeypatch):
    captured_calls: list[tuple[str, dict]] = []

    monkeypatch.setattr(
        final_diagnosis_node_module,
        "build_llm_agent",
        lambda **_kwargs: (object(), "final-model"),
    )
    monkeypatch.setattr(
        final_diagnosis_node_module,
        "agent_invoke",
        lambda _agent, _messages: SimpleNamespace(
            payload={
                "messages": [
                    SimpleNamespace(
                        content=(
                            '{"diagnosis_text": "结合你的情况，更像是常见感冒方向。", '
                            '"should_recommend_products": true, '
                            '"product_keyword": "感冒药", '
                            '"product_usage": "缓解咳嗽和流鼻涕"}'
                        )
                    )
                ]
            },
            content="",
        ),
    )
    monkeypatch.setattr(
        final_diagnosis_node_module,
        "record_agent_trace",
        lambda **_kwargs: {
            "text": "结合你的情况，更像是常见感冒方向。",
            "is_usage_complete": True,
            "usage": None,
        },
    )

    def _fake_invoke(tool_object, payload):
        captured_calls.append((tool_object.name, payload))
        if tool_object.name == "search_products":
            return {
                "rows": [
                    {"id": 101, "name": "感冒灵颗粒"},
                    {"id": 102, "name": "板蓝根颗粒"},
                ]
            }
        return "__SUCCESS__"

    monkeypatch.setattr(final_diagnosis_node_module, "invoke_runnable", _fake_invoke)

    result = final_diagnosis_node_module.consultation_final_diagnosis_node(
        {
            "history_messages": [HumanMessage(content="我咳嗽两天，还有点流鼻涕")],
            "task_difficulty": "normal",
        }
    )

    assert result["consultation_status"] == CONSULTATION_STATUS_COMPLETED
    assert result["recommended_product_ids"] == [101, 102]
    assert "感冒灵颗粒" in result["final_text"]
    assert "板蓝根颗粒" in result["final_text"]
    assert captured_calls == [
        (
            "search_products",
            {
                "keyword": "感冒药",
                "usage": "缓解咳嗽和流鼻涕",
                "page_num": 1,
                "page_size": 3,
            },
        ),
        (
            "send_product_purchase_card",
            {
                "items": [
                    {"productId": 101, "quantity": 1},
                    {"productId": 102, "quantity": 1},
                ]
            },
        ),
    ]
    assert [item["tool_name"] for item in result["diagnosis_trace"]["tool_calls"]] == [
        "search_products",
        "send_product_purchase_card",
    ]


def test_consultation_final_diagnosis_node_skips_purchase_card_when_search_empty(monkeypatch):
    captured_calls: list[tuple[str, dict]] = []

    monkeypatch.setattr(
        final_diagnosis_node_module,
        "build_llm_agent",
        lambda **_kwargs: (object(), "final-model"),
    )
    monkeypatch.setattr(
        final_diagnosis_node_module,
        "agent_invoke",
        lambda _agent, _messages: SimpleNamespace(
            payload={
                "messages": [
                    SimpleNamespace(
                        content=(
                            '{"diagnosis_text": "更像是轻度咽喉不适，建议先观察。", '
                            '"should_recommend_products": true, '
                            '"product_keyword": "咽喉药", '
                            '"product_usage": "缓解咽喉不适"}'
                        )
                    )
                ]
            },
            content="",
        ),
    )
    monkeypatch.setattr(
        final_diagnosis_node_module,
        "record_agent_trace",
        lambda **_kwargs: {
            "text": "更像是轻度咽喉不适，建议先观察。",
            "is_usage_complete": True,
            "usage": None,
        },
    )

    def _fake_invoke(tool_object, payload):
        captured_calls.append((tool_object.name, payload))
        if tool_object.name == "search_products":
            return {"rows": []}
        raise AssertionError("search empty 时不应继续发送购买卡")

    monkeypatch.setattr(final_diagnosis_node_module, "invoke_runnable", _fake_invoke)

    result = final_diagnosis_node_module.consultation_final_diagnosis_node(
        {
            "history_messages": [HumanMessage(content="我喉咙有点不舒服")],
            "task_difficulty": "normal",
        }
    )

    assert result["recommended_product_ids"] == []
    assert result["final_text"] == "更像是轻度咽喉不适，建议先观察。"
    assert captured_calls == [
        (
            "search_products",
            {
                "keyword": "咽喉药",
                "usage": "缓解咽喉不适",
                "page_num": 1,
                "page_size": 3,
            },
        )
    ]
