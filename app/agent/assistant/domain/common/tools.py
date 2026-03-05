"""
管理后台通用工具与非订单/商品工具。

约定：
1. 所有批量 ID 参数必须传 JSON 数组（List[str]），不能传逗号拼接字符串；
2. 所有参数命名以工具函数签名为准，Agent 调用时必须传同名字段；
3. 详情类工具在收到空 ID 或全空白 ID 时会直接报错并拒绝调用后端接口。
"""

import datetime

from langchain_core.tools import tool

from app.agent.assistant.domain.common.schema import UserInfo
from app.core.agent.agent_tool_events import tool_call_status
from app.core.security import get_current_user


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
    description="获取当前聊天用户的基本信息。"
                "何时使用？ 当用户需要获取自己的基本信息时。或者需要查询当前用户的个人信息时，"
)
@tool_call_status(
    tool_name="获取用户信息",
    start_message="正在获取用户信息",
    error_message="获取用户信息失败",
    timely_message="用户信息正在持续处理中",
)
def get_safe_user_info() -> UserInfo:
    """获取当前登录用户的基本信息（已过滤敏感信息）。"""
    auth_user = get_current_user()
    return UserInfo.from_auth_user(auth_user)


@tool(
    description="获取当前系统时间（ISO 8601 格式）获取当前的时间必须调用此参数。"
)
@tool_call_status(
    tool_name="获取当前时间",
    start_message="正在获取当前时间",
    error_message="获取当前时间失败",
    timely_message="当前时间正在持续处理中",
)
def get_current_time() -> dict:
    """返回当前系统时间。"""

    now = datetime.datetime.now(datetime.timezone.utc)
    return {
        "current_time": now.isoformat(),
        "timezone": "UTC",
    }


ADMIN_TOOLS = [
    get_safe_user_info,
    get_current_time,
]
