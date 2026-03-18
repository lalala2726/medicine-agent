import asyncio
from uuid import UUID

import pytest
from pydantic import ValidationError

from app.agent.client.domain.after_sale import tools as after_sale_tools_module
from app.agent.client.domain.after_sale.schema import AfterSaleEligibilityRequest
from app.agent.client.domain.common.frontend_card_tools import (
    SendProductCardRequest,
    send_product_card,
)
from app.agent.client.domain.common.user_action_tools import (
    OpenUserAfterSaleListRequest,
    OpenUserOrderListRequest,
    open_user_after_sale_list,
    open_user_order_list,
)
from app.agent.client.domain.order import tools as order_tools_module
from app.agent.client.domain.product import tools as product_tools_module
from app.agent.client.domain.product.schema import ProductSearchRequest
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


def test_open_user_after_sale_list_enqueues_action_with_status():
    queue_token = set_final_response_queue()
    try:
        result = asyncio.run(
            open_user_after_sale_list.ainvoke({"afterSaleStatus": "PENDING"})
        )
        queued_responses = drain_final_sse_responses()
    finally:
        reset_final_response_queue(queue_token)

    assert result == "已为你打开待审核售后列表"
    assert len(queued_responses) == 1
    response = queued_responses[0]
    assert response.type == MessageType.ACTION
    assert response.content.message == "已为你打开待审核售后列表"
    assert response.action is not None
    assert response.action.target == "user_after_sale_list"
    assert response.action.payload.afterSaleStatus == "PENDING"
    assert response.action.priority == 100


def test_open_user_after_sale_list_request_rejects_invalid_status():
    with pytest.raises(ValidationError):
        OpenUserAfterSaleListRequest.model_validate({"afterSaleStatus": "INVALID"})


def test_send_product_card_enqueues_card_response():
    queue_token = set_final_response_queue()
    try:
        result = asyncio.run(send_product_card.ainvoke({"productIds": [1001, 1002]}))
        queued_responses = drain_final_sse_responses()
    finally:
        reset_final_response_queue(queue_token)

    assert result == "已准备商品卡片"
    assert len(queued_responses) == 1
    response = queued_responses[0]
    assert response.type == MessageType.CARD
    assert response.card is not None
    assert response.card.type == "product-card"
    assert response.card.version == 1
    assert response.card.data.productIds == [1001, 1002]
    assert response.content.model_dump(exclude_none=True) == {}
    assert response.meta is not None
    assert "card_uuid" in response.meta
    assert str(UUID(response.meta["card_uuid"])) == response.meta["card_uuid"]


def test_send_product_card_request_rejects_empty_product_ids():
    with pytest.raises(ValidationError):
        SendProductCardRequest.model_validate({"productIds": []})


def test_send_product_card_request_rejects_non_positive_product_id():
    with pytest.raises(ValidationError):
        SendProductCardRequest.model_validate({"productIds": [0]})


def test_product_search_request_requires_query():
    with pytest.raises(ValidationError):
        ProductSearchRequest.model_validate({})


def test_after_sale_eligibility_request_rejects_blank_order_no():
    with pytest.raises(ValidationError):
        AfterSaleEligibilityRequest.model_validate({"order_no": "   "})


def test_get_order_detail_calls_client_order_detail_endpoint(monkeypatch):
    captured: dict = {}

    class _FakeHttpClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, *, url, params=None):
            captured["url"] = url
            captured["params"] = params
            return {"ok": True}

    monkeypatch.setattr(order_tools_module, "HttpClient", _FakeHttpClient)
    monkeypatch.setattr(order_tools_module.HttpResponse, "parse_data", lambda response: response)

    result = asyncio.run(
        order_tools_module.get_order_detail.ainvoke({"order_no": "O202603160001"})
    )

    assert captured == {
        "url": "/agent/client/order/O202603160001",
        "params": None,
    }
    assert result == {"ok": True}


def test_search_products_calls_client_search_endpoint(monkeypatch):
    captured: dict = {}

    class _FakeHttpClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, *, url, params=None):
            captured["url"] = url
            captured["params"] = params
            return {"rows": []}

    monkeypatch.setattr(product_tools_module, "HttpClient", _FakeHttpClient)
    monkeypatch.setattr(
        product_tools_module.HttpResponse,
        "parse_data",
        lambda response: response,
    )

    result = asyncio.run(
        product_tools_module.search_products.ainvoke({"keyword": "维生素"})
    )

    assert captured == {
        "url": "/agent/client/product/search",
        "params": {
            "keyword": "维生素",
            "categoryName": None,
            "usage": None,
            "pageNum": 1,
            "pageSize": 10,
        },
    }
    assert result == {"rows": []}


def test_check_after_sale_eligibility_calls_client_endpoint(monkeypatch):
    captured: dict = {}

    class _FakeHttpClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, *, url, params=None):
            captured["url"] = url
            captured["params"] = params
            return {"eligible": True}

    monkeypatch.setattr(after_sale_tools_module, "HttpClient", _FakeHttpClient)
    monkeypatch.setattr(
        after_sale_tools_module.HttpResponse,
        "parse_data",
        lambda response: response,
    )

    result = asyncio.run(
        after_sale_tools_module.check_after_sale_eligibility.ainvoke(
            {"order_no": "O202603160001", "order_item_id": 12}
        )
    )

    assert captured == {
        "url": "/agent/client/after-sale/eligibility",
        "params": {
            "orderNo": "O202603160001",
            "orderItemId": 12,
        },
    }
    assert result == {"eligible": True}
