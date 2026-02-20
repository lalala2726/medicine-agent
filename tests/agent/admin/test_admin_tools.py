import asyncio

import pytest

from app.agent.admin.tools import admin_tools


def test_normalize_id_list_rejects_empty_values():
    """测试目标：空 ID 列表应被拒绝，避免拼接出空路径。"""

    with pytest.raises(ValueError):
        admin_tools._normalize_id_list(["", "   "], field_name="order_id")


def test_get_orders_detail_rejects_empty_id_list():
    """测试目标：get_orders_detail 在空 ID 时直接报错，不访问 /agent/order/。"""

    with pytest.raises(ValueError):
        asyncio.run(admin_tools.get_orders_detail.coroutine(order_id=["", " "]))
