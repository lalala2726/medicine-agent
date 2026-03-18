"""Client 商品域节点包。"""

from app.agent.client.domain.product.node import product_agent
from app.agent.client.domain.product.tools import (
    get_product_detail,
    get_product_spec,
    search_products,
)

__all__ = [
    "get_product_detail",
    "get_product_spec",
    "product_agent",
    "search_products",
]
