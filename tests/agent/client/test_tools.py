import asyncio

import pytest
from pydantic import ValidationError

from app.agent.client.tools import OpenUserOrderListRequest, open_user_order_list
from app.core.agent.agent_event_bus import (
    drain_final_sse_responses,
    reset_final_response_queue,
    set_final_response_queue,
)
from app.schemas.sse_response import MessageType


def test_open_user_order_list_enqueues_action_with_order_status():
    queue_token = set_final_response_queue()
    try:
        result = asyncio.run(open_user_order_list.ainvoke({"orderStatus": "PENDING_PAYMENT"}))
        queued_responses = drain_final_sse_responses()
    finally:
        reset_final_response_queue(queue_token)

    assert result == "已为你打开待支付订单列表"
    assert len(queued_responses) == 1
    response = queued_responses[0]
    assert response.type == MessageType.ACTION
    assert response.content.message == "已为你打开待支付订单列表"
    assert response.action is not None
    assert response.action.target == "user_order_list"
    assert response.action.payload.orderStatus == "PENDING_PAYMENT"
    assert response.action.priority == 100


def test_open_user_order_list_enqueues_action_without_order_status():
    queue_token = set_final_response_queue()
    try:
        result = asyncio.run(open_user_order_list.ainvoke({}))
        queued_responses = drain_final_sse_responses()
    finally:
        reset_final_response_queue(queue_token)

    assert result == "已为你打开订单列表"
    assert len(queued_responses) == 1
    response = queued_responses[0]
    assert response.action is not None
    assert response.action.target == "user_order_list"
    assert response.action.payload.orderStatus is None


def test_open_user_order_list_request_rejects_invalid_status():
    with pytest.raises(ValidationError):
        OpenUserOrderListRequest.model_validate({"orderStatus": "INVALID"})
