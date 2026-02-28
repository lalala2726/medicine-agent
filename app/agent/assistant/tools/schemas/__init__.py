"""
`app.agent.assistant.tools.schemas` schema 导出模块。

按工具域拆分输入/输出模型，避免工具函数与参数模型混杂在同一文件。
"""

from app.agent.assistant.tools.schemas.after_sale import (
    AdminAfterSaleIdRequest,
    AdminAfterSaleListQueryRequest,
)
from app.agent.assistant.tools.schemas.analytics import (
    AnalyticsOrderTrendRequest,
    AnalyticsTopLimitRequest,
)
from app.agent.assistant.tools.schemas.base import UserInfo
from app.agent.assistant.tools.schemas.order import (
    MallOrderListRequest,
    OrderDetailRequest,
    OrderIdRequest,
)
from app.agent.assistant.tools.schemas.product import (
    DrugDetailRequest,
    MallProductListQueryRequest,
    ProductInfoRequest,
)
from app.agent.assistant.tools.schemas.user import (
    AdminUserIdPageRequest,
    AdminUserIdRequest,
    AdminUserListQueryRequest,
)

__all__ = [
    "AdminAfterSaleIdRequest",
    "AdminAfterSaleListQueryRequest",
    "AnalyticsOrderTrendRequest",
    "AnalyticsTopLimitRequest",
    "DrugDetailRequest",
    "MallOrderListRequest",
    "MallProductListQueryRequest",
    "OrderDetailRequest",
    "OrderIdRequest",
    "ProductInfoRequest",
    "UserInfo",
    "AdminUserIdPageRequest",
    "AdminUserIdRequest",
    "AdminUserListQueryRequest",
]
