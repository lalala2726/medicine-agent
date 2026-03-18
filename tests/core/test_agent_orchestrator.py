import asyncio
import json
from types import SimpleNamespace
from uuid import UUID

from app.core.agent import agent_orchestrator as orchestrator_module
from app.core.agent.agent_event_bus import enqueue_final_sse_response
from app.schemas.sse_response import (
    Action,
    AssistantResponse,
    Content,
    MessageType,
    ProductCard,
    ProductCardData,
    UserOrderListPayload,
)


def _parse_sse_payload(raw_event: str) -> dict:
    prefix = "data: "
    assert raw_event.startswith(prefix)
    return json.loads(raw_event[len(prefix):].strip())


def _build_action_response(*, message: str, order_status: str | None, priority: int) -> AssistantResponse:
    return AssistantResponse(
        type=MessageType.ACTION,
        content=Content(message=message),
        action=Action(
            type="navigate",
            target="user_order_list",
            payload=UserOrderListPayload(orderStatus=order_status),
            priority=priority,
        ),
    )


def _build_card_response(*product_ids: int) -> AssistantResponse:
    return AssistantResponse(
        type=MessageType.CARD,
        content=Content(),
        card=ProductCard(
            data=ProductCardData(productIds=list(product_ids)),
        ),
        meta={
            "card_uuid": "123e4567-e89b-12d3-a456-426614174000",
        },
    )


def test_serialize_sse_includes_action_payload():
    payload = _build_action_response(
        message="已为你打开待支付订单列表",
        order_status="PENDING_PAYMENT",
        priority=100,
    )

    serialized = orchestrator_module.serialize_sse(payload)
    parsed = _parse_sse_payload(serialized)

    assert parsed["type"] == "action"
    assert parsed["content"]["message"] == "已为你打开待支付订单列表"
    assert parsed["action"] == {
        "type": "navigate",
        "target": "user_order_list",
        "payload": {"orderStatus": "PENDING_PAYMENT"},
        "priority": 100,
    }


def test_event_stream_flushes_final_actions_by_priority_before_end():
    class _FakeWorkflow:
        async def astream(self, _state: dict, **_kwargs):
            yield ("messages", (SimpleNamespace(content="你好"), {"langgraph_node": "chat_agent"}))
            enqueue_final_sse_response(
                _build_action_response(
                    message="已为你打开已完成订单列表",
                    order_status="COMPLETED",
                    priority=50,
                )
            )
            enqueue_final_sse_response(
                _build_action_response(
                    message="已为你打开待支付订单列表",
                    order_status="PENDING_PAYMENT",
                    priority=100,
                )
            )
            enqueue_final_sse_response(
                _build_action_response(
                    message="已为你打开待收货订单列表",
                    order_status="PENDING_RECEIPT",
                    priority=50,
                )
            )

    config = orchestrator_module.AssistantStreamConfig(
        workflow=_FakeWorkflow(),
        build_initial_state=lambda _question: {},
        extract_final_content=lambda _state: "",
        should_stream_token=lambda _node, _state: True,
        build_stream_config=None,
        invoke_sync=lambda state: state,
        map_exception=lambda exc: str(exc),
    )

    async def _collect() -> list[str]:
        events: list[str] = []
        async for event in orchestrator_module._event_stream(question="你好", config=config):
            events.append(event)
        return events

    rendered_events = asyncio.run(_collect())
    payloads = [_parse_sse_payload(event) for event in rendered_events]

    assert [payload["type"] for payload in payloads] == [
        "answer",
        "action",
        "action",
        "action",
        "answer",
    ]
    assert payloads[0]["content"]["text"] == "你好"
    assert payloads[1]["action"]["payload"]["orderStatus"] == "PENDING_PAYMENT"
    assert payloads[2]["action"]["payload"]["orderStatus"] == "COMPLETED"
    assert payloads[3]["action"]["payload"]["orderStatus"] == "PENDING_RECEIPT"
    assert payloads[4]["is_end"] is True


def test_serialize_sse_includes_product_card_payload():
    payload = _build_card_response(1001, 1002)

    serialized = orchestrator_module.serialize_sse(payload)
    parsed = _parse_sse_payload(serialized)

    assert parsed["type"] == "card"
    assert parsed["content"] == {}
    assert parsed["card"] == {
        "type": "product-card",
        "version": 1,
        "data": {
            "productIds": [1001, 1002],
        },
    }
    assert parsed["meta"] == {
        "card_uuid": "123e4567-e89b-12d3-a456-426614174000",
    }


def test_event_stream_flushes_final_card_before_end():
    class _FakeWorkflow:
        async def astream(self, _state: dict, **_kwargs):
            yield ("messages", (SimpleNamespace(content="你好"), {"langgraph_node": "chat_agent"}))
            enqueue_final_sse_response(_build_card_response(1001, 1002))

    config = orchestrator_module.AssistantStreamConfig(
        workflow=_FakeWorkflow(),
        build_initial_state=lambda _question: {},
        extract_final_content=lambda _state: "",
        should_stream_token=lambda _node, _state: True,
        build_stream_config=None,
        invoke_sync=lambda state: state,
        map_exception=lambda exc: str(exc),
    )

    async def _collect() -> list[str]:
        events: list[str] = []
        async for event in orchestrator_module._event_stream(question="你好", config=config):
            events.append(event)
        return events

    rendered_events = asyncio.run(_collect())
    payloads = [_parse_sse_payload(event) for event in rendered_events]

    assert [payload["type"] for payload in payloads] == [
        "answer",
        "card",
        "answer",
    ]
    assert payloads[1]["card"]["data"]["productIds"] == [1001, 1002]
    assert str(UUID(payloads[1]["meta"]["card_uuid"])) == payloads[1]["meta"]["card_uuid"]
    assert payloads[2]["is_end"] is True


def test_event_stream_keeps_action_priority_ahead_of_card():
    class _FakeWorkflow:
        async def astream(self, _state: dict, **_kwargs):
            yield ("messages", (SimpleNamespace(content="你好"), {"langgraph_node": "chat_agent"}))
            enqueue_final_sse_response(_build_card_response(1001, 1002))
            enqueue_final_sse_response(
                _build_action_response(
                    message="已为你打开待支付订单列表",
                    order_status="PENDING_PAYMENT",
                    priority=100,
                )
            )
            enqueue_final_sse_response(
                _build_action_response(
                    message="已为你打开已完成订单列表",
                    order_status="COMPLETED",
                    priority=50,
                )
            )

    config = orchestrator_module.AssistantStreamConfig(
        workflow=_FakeWorkflow(),
        build_initial_state=lambda _question: {},
        extract_final_content=lambda _state: "",
        should_stream_token=lambda _node, _state: True,
        build_stream_config=None,
        invoke_sync=lambda state: state,
        map_exception=lambda exc: str(exc),
    )

    async def _collect() -> list[str]:
        events: list[str] = []
        async for event in orchestrator_module._event_stream(question="你好", config=config):
            events.append(event)
        return events

    rendered_events = asyncio.run(_collect())
    payloads = [_parse_sse_payload(event) for event in rendered_events]

    assert [payload["type"] for payload in payloads] == [
        "answer",
        "action",
        "action",
        "card",
        "answer",
    ]
    assert payloads[1]["action"]["payload"]["orderStatus"] == "PENDING_PAYMENT"
    assert payloads[2]["action"]["payload"]["orderStatus"] == "COMPLETED"
    assert payloads[3]["card"]["data"]["productIds"] == [1001, 1002]
    assert str(UUID(payloads[3]["meta"]["card_uuid"])) == payloads[3]["meta"]["card_uuid"]
