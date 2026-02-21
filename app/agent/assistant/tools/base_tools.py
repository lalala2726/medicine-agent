"""
管理后台通用工具与非订单/商品工具。

约定：
1. 所有批量 ID 参数必须传 JSON 数组（List[str]），不能传逗号拼接字符串；
2. 所有参数命名以工具函数签名为准，Agent 调用时必须传同名字段；
3. 详情类工具在收到空 ID 或全空白 ID 时会直接报错并拒绝调用后端接口。
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.core.assistant_status import tool_call_status
from app.schemas.http_response import HttpResponse
from app.utils.http_client import HttpClient


def format_ids_to_string(ids: list[str]) -> str:
    """将 ID 列表转换为逗号分隔字符串。"""

    return ",".join(str(id_) for id_ in ids)


def _normalize_id_list(ids: list[str], *, field_name: str) -> list[str]:
    """规范化并校验批量 ID 参数。"""

    normalized = [str(item).strip() for item in ids if str(item).strip()]
    if not normalized:
        raise ValueError(f"{field_name} 必须为非空字符串数组（List[str]），例如 [\"A1\",\"A2\"]")
    return normalized


class DrugDetailRequest(BaseModel):
    """
    药品详情查询请求参数。

    传参示例：
    `{"product_id": ["2001", "2003"]}`
    """

    product_id: list[str] = Field(
        min_length=1,
        description=(
            "药品商品ID字符串数组（List[str]），支持批量查询。"
            "必须传 JSON 数组，不能传逗号拼接字符串。"
        ),
        examples=[["2001"], ["2001", "2003"]],
    )


@tool(
    description="获取当前登录用户的基本信息。"
    "调用时机：用户询问「我是谁」「我的账户信息」，或需要用户ID进行后续操作时。"
)
@tool_call_status(
    tool_name="获取用户信息",
    start_message="正在获取用户信息",
    error_message="获取用户信息失败",
    timely_message="用户信息正在持续处理中",
)
async def get_user_info() -> dict:
    """获取当前登录用户的基本信息。"""

    async with HttpClient() as client:
        response = await client.get(url="/agent/info")
        return HttpResponse.parse_data(response)


@tool(
    args_schema=DrugDetailRequest,
    description=(
        "根据商品ID获取药品详细信息，包括说明书、适应症、用法用量等，支持批量查询。"
        "参数传递规则：product_id 必须是字符串数组 List[str]，例如 "
        "{\"product_id\": [\"2001\", \"2003\"]}；"
        "不要传逗号拼接字符串。"
        "调用时机：用户询问药品说明书、适应症、用法用量等信息时。"
    ),
)
@tool_call_status(
    tool_name="获取药品详情",
    start_message="正在查询药品详情",
    error_message="获取药品详情失败",
    timely_message="药品详情正在持续处理中",
)
async def get_drug_detail(product_id: list[str]) -> dict:
    """根据药品商品ID获取详细药品信息。"""

    normalized_ids = _normalize_id_list(product_id, field_name="product_id")
    ids_str = format_ids_to_string(normalized_ids)
    async with HttpClient() as client:
        response = await client.get(url=f"/agent/drug/{ids_str}")
        return HttpResponse.parse_data(response)


ADMIN_TOOLS = [
    get_user_info,
    get_drug_detail,
]
