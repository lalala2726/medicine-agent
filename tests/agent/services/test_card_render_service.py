import asyncio

import pytest

from app.agent.services import card_render_service as card_render_service_module
from app.core.exception.exceptions import ServiceException


def test_render_product_card_requests_purchase_cards_endpoint(monkeypatch):
    captured: dict = {}

    class _FakeHttpClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, **kwargs):
            captured["url"] = url
            captured["kwargs"] = kwargs
            return {
                "totalPrice": "36.70",
                "items": [
                    {
                        "id": "101",
                        "name": "布洛芬缓释胶囊",
                        "image": "https://example.com/101.png",
                        "price": "16.80",
                        "spec": "24粒/盒",
                        "efficacy": "缓解发热、头痛",
                        "prescription": False,
                        "stock": 56,
                    },
                    {
                        "id": "102",
                        "name": "维生素C咀嚼片",
                        "image": "https://example.com/102.png",
                        "price": "19.90",
                        "spec": "60片/瓶",
                        "efficacy": "补充维生素C",
                        "prescription": False,
                        "stock": 98,
                    },
                ],
                "meta": {
                    "entityDescription": "客户端智能体商品购买卡片",
                    "fieldDescriptions": {
                        "items[].id": "商品ID",
                    },
                },
            }

    monkeypatch.setattr(card_render_service_module, "HttpClient", _FakeHttpClient)

    card = asyncio.run(card_render_service_module.render_product_card([102, 101]))

    assert captured == {
        "url": "/agent/client/purchase_cards/102,101",
        "kwargs": {
            "response_format": "json",
        },
    }
    assert card is not None
    assert card.type == "product-card"
    assert card.data == {
        "title": "为您推荐以下商品",
        "products": [
            {
                "id": "102",
                "name": "维生素C咀嚼片",
                "image": "https://example.com/102.png",
                "price": "19.90",
            },
            {
                "id": "101",
                "name": "布洛芬缓释胶囊",
                "image": "https://example.com/101.png",
                "price": "16.80",
            },
        ],
    }


def test_render_product_card_returns_none_when_items_are_empty(monkeypatch):
    class _FakeHttpClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, **kwargs):
            return {
                "totalPrice": "0.00",
                "items": [],
                "meta": {
                    "entityDescription": "客户端智能体商品购买卡片",
                    "fieldDescriptions": {},
                },
            }

    monkeypatch.setattr(card_render_service_module, "HttpClient", _FakeHttpClient)

    card = asyncio.run(card_render_service_module.render_product_card([102, 101]))

    assert card is None


def test_render_product_card_propagates_service_exception(monkeypatch):
    class _FakeHttpClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, **kwargs):
            raise ServiceException(code=503, message="业务端暂不可用")

    monkeypatch.setattr(card_render_service_module, "HttpClient", _FakeHttpClient)

    with pytest.raises(ServiceException, match="业务端暂不可用"):
        asyncio.run(card_render_service_module.render_product_card([102, 101]))
