"""
管理后台通用工具与非订单/商品工具。

约定：
1. 所有批量 ID 参数必须传 JSON 数组（List[str]），不能传逗号拼接字符串；
2. 所有参数命名以工具函数签名为准，Agent 调用时必须传同名字段；
3. 详情类工具在收到空 ID 或全空白 ID 时会直接报错并拒绝调用后端接口。
"""

from langchain_core.tools import tool
from app.core.sse_tool_events import tool_call_status
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


ADMIN_TOOLS = [
    get_user_info,
]
