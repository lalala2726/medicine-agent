"""
管理后台通用工具与非订单/商品工具。

约定：
1. 所有批量 ID 参数必须传 JSON 数组（List[str]），不能传逗号拼接字符串；
2. 所有参数命名以工具函数签名为准，Agent 调用时必须传同名字段；
3. 详情类工具在收到空 ID 或全空白 ID 时会直接报错并拒绝调用后端接口。
"""

import datetime

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agent.assistant.domain.common.schema import UserInfo
from app.core.agent.agent_tool_events import tool_call_status
from app.core.security import get_current_user
from app.rag import (
    format_knowledge_search_hits,
    query_knowledge_by_rewritten_question,
)


def format_ids_to_string(ids: list[str]) -> str:
    """将 ID 列表转换为逗号分隔字符串。"""

    return ",".join(str(id_) for id_ in ids)


def _normalize_id_list(ids: list[str], *, field_name: str) -> list[str]:
    """规范化并校验批量 ID 参数。"""

    normalized = [str(item).strip() for item in ids if str(item).strip()]
    if not normalized:
        raise ValueError(f"{field_name} 必须为非空字符串数组（List[str]），例如 [\"A1\",\"A2\"]")
    return normalized


class KnowledgeSearchToolRequest(BaseModel):
    """知识库检索工具的入参模型。

    Attributes:
        query: 用于检索的原始用户问题。
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="用户原始问题")

    @field_validator("query")
    @classmethod
    def _normalize_query(cls, value: str) -> str:
        """在字段校验通过前规范化 query。

        Args:
            value: tool 调用时传入的原始查询文本。

        Returns:
            去除首尾空白后的查询文本。

        Raises:
            ValueError: 当 query 去空白后为空时抛出。
        """

        normalized = value.strip()
        if not normalized:
            raise ValueError("query 不能为空")
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


@tool(
    args_schema=KnowledgeSearchToolRequest,
    description=(
            "检索固定知识库中的相关文档片段。"
            "适用于制度说明、文档知识、产品资料、FAQ、规则解释等问题。"
            "先检索再回答，不要编造未检索到的知识内容。直接传入用户的原始问题即可，不要添加其他文字。"
    ),
)
@tool_call_status(
    tool_name="知识库检索",
    start_message="正在检索知识库",
    error_message="知识库检索失败",
    timely_message="知识库检索正在持续处理中",
)
def search_knowledge_context(query: str) -> str:
    """检索知识库并返回格式化后的上下文文本。

    Args:
        query: 需要先改写再检索的原始用户问题。

    Returns:
        基于改写检索命中结果拼接得到的格式化文本块。
    """

    hits = query_knowledge_by_rewritten_question(
        question=query,
        top_k=None,
    )
    return format_knowledge_search_hits(hits)


ADMIN_TOOLS = [
    search_knowledge_context,
    get_safe_user_info,
    get_current_time,
]
