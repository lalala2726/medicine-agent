"""
管理端基础工具与共享辅助函数。

说明：
1. 基础工具默认直接暴露给 admin 单 Agent；
2. 业务工具授权逻辑也从本模块提供，避免节点层分散控制；
3. 共享的 ID 规范化函数放在此处，供各领域工具复用。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agent.admin.state import AgentState
from app.core.agent.agent_tool_events import tool_call_status
from app.core.security import get_current_user
from app.rag import (
    format_knowledge_search_hits,
    query_knowledge_by_rewritten_question,
)
from app.schemas.auth import AuthUser


def format_ids_to_string(ids: list[str]) -> str:
    """
    功能描述：
        将字符串 ID 数组按逗号拼接为后端接口所需路径片段。

    参数说明：
        ids (list[str]): 已完成校验的 ID 数组。

    返回值：
        str: 逗号拼接后的字符串。

    异常说明：
        无。
    """

    return ",".join(str(item) for item in ids)


def normalize_id_list(ids: list[str], *, field_name: str) -> list[str]:
    """
    功能描述：
        规范化并校验批量 ID 数组参数。

    参数说明：
        ids (list[str]): 原始 ID 数组。
        field_name (str): 当前字段名，用于错误提示。

    返回值：
        list[str]: 去空白、去空项后的 ID 数组。

    异常说明：
        ValueError: 当数组为空或全部为空白字符串时抛出。
    """

    normalized = [str(item).strip() for item in ids if str(item).strip()]
    if not normalized:
        raise ValueError(f"{field_name} 必须为非空字符串数组（List[str]）")
    return normalized


def merge_unique_tool_keys(
        existing_tool_keys: list[str],
        requested_tool_keys: list[str],
) -> list[str]:
    """
    功能描述：
        合并已授权工具与本次申请工具，并保持顺序去重。

    参数说明：
        existing_tool_keys (list[str]): 当前状态中已有的工具 key 数组。
        requested_tool_keys (list[str]): 本次新申请的工具 key 数组。

    返回值：
        list[str]: 合并后的稳定顺序工具 key 数组。

    异常说明：
        无。
    """

    merged_tool_keys: list[str] = []
    for raw_tool_key in [*existing_tool_keys, *requested_tool_keys]:
        tool_key = str(raw_tool_key or "").strip()
        if not tool_key:
            continue
        if tool_key in merged_tool_keys:
            continue
        merged_tool_keys.append(tool_key)
    return merged_tool_keys


class UserInfo(BaseModel):
    """
    功能描述：
        当前登录用户的安全信息模型，仅暴露允许给 Agent 使用的字段。

    参数说明：
        username (str | None): 用户名。
        nickname (str | None): 昵称。

    返回值：
        无（数据模型定义）。

    异常说明：
        无。
    """

    username: str | None = None
    nickname: str | None = None

    @classmethod
    def from_auth_user(cls, auth_user: AuthUser) -> "UserInfo":
        """
        功能描述：
            从认证用户对象构造安全用户信息对象。

        参数说明：
            auth_user (AuthUser): 当前认证用户对象。

        返回值：
            UserInfo: 仅包含非敏感字段的用户信息。

        异常说明：
            无。
        """

        return cls(
            username=auth_user.username,
            nickname=auth_user.nickname,
        )


class KnowledgeSearchToolRequest(BaseModel):
    """
    功能描述：
        知识库检索工具入参模型。

    参数说明：
        query (str): 用户原始问题。

    返回值：
        无（数据模型定义）。

    异常说明：
        ValueError: 当 query 去空白后为空时抛出。
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="用户原始问题")

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        """
        功能描述：
            规范化知识检索问题文本。

        参数说明：
            value (str): 原始问题文本。

        返回值：
            str: 去除首尾空白后的问题文本。

        异常说明：
            ValueError: 当文本为空时抛出。
        """

        normalized = value.strip()
        if not normalized:
            raise ValueError("query 不能为空")
        return normalized


class RequestToolAccessRequest(BaseModel):
    """
    功能描述：
        工具授权申请入参模型。

    参数说明：
        tool_keys (list[str]): 本次申请的业务工具 key 数组。
        reason (str): 申请原因。

    返回值：
        无（数据模型定义）。

    异常说明：
        ValueError: 当工具数组或原因不合法时抛出。
    """

    model_config = ConfigDict(extra="forbid")

    tool_keys: list[str] = Field(
        min_length=1,
        description="需要授权的业务工具 key 数组，只允许 snake_case 工具名",
    )
    reason: str = Field(
        ...,
        min_length=1,
        description="本次申请工具的业务原因说明",
    )

    @field_validator("tool_keys")
    @classmethod
    def normalize_tool_keys(cls, value: list[str]) -> list[str]:
        """
        功能描述：
            规范化工具 key 数组并按顺序去重。

        参数说明：
            value (list[str]): 原始工具 key 数组。

        返回值：
            list[str]: 去空白、转小写、顺序去重后的工具 key 数组。

        异常说明：
            ValueError: 当数组为空或包含空值时抛出。
        """

        normalized_tool_keys: list[str] = []
        for raw_tool_key in value:
            tool_key = str(raw_tool_key or "").strip().lower()
            if not tool_key:
                raise ValueError("tool_keys 不能包含空值")
            if tool_key in normalized_tool_keys:
                continue
            normalized_tool_keys.append(tool_key)
        if not normalized_tool_keys:
            raise ValueError("tool_keys 不能为空")
        return normalized_tool_keys

    @field_validator("reason")
    @classmethod
    def normalize_reason(cls, value: str) -> str:
        """
        功能描述：
            规范化工具申请原因文本。

        参数说明：
            value (str): 原始申请原因。

        返回值：
            str: 去除首尾空白后的原因文本。

        异常说明：
            ValueError: 当原因为空时抛出。
        """

        normalized = value.strip()
        if not normalized:
            raise ValueError("reason 不能为空")
        return normalized


@tool(
    description=(
            "获取当前聊天用户的基本信息。"
            "适用于确认当前登录人身份、昵称等基础上下文。"
    ),
)
@tool_call_status(
    tool_name="获取用户信息",
    start_message="正在获取用户信息",
    error_message="获取用户信息失败",
    timely_message="用户信息正在持续处理中",
)
def get_safe_user_info() -> UserInfo:
    """
    功能描述：
        获取当前登录用户的非敏感基础信息。

    参数说明：
        无。

    返回值：
        UserInfo: 过滤敏感字段后的用户信息。

    异常说明：
        无。
    """

    auth_user = get_current_user()
    return UserInfo.from_auth_user(auth_user)


@tool(
    args_schema=KnowledgeSearchToolRequest,
    description=(
            "检索固定知识库中的相关文档片段。"
            "适用于制度说明、文档知识、产品资料、FAQ、规则解释等问题。"
            "直接传入用户原始问题，不要拼接额外提示语。"
    ),
)
@tool_call_status(
    tool_name="知识库检索",
    start_message="正在检索知识库",
    error_message="知识库检索失败",
    timely_message="知识库检索正在持续处理中",
)
def search_knowledge_context(query: str) -> str:
    """
    功能描述：
        对知识库执行问题改写检索，并返回格式化后的知识片段文本。

    参数说明：
        query (str): 用户原始问题。

    返回值：
        str: 知识检索结果的格式化文本。

    异常说明：
        无。
    """

    hits = query_knowledge_by_rewritten_question(
        question=query,
        top_k=None,
    )
    return format_knowledge_search_hits(hits)


def create_request_tool_access_tool(
        *,
        get_allowed_tool_keys: Callable[[], set[str]],
) -> Any:
    """
    功能描述：
        创建运行时工具授权申请工具。

    参数说明：
        get_allowed_tool_keys (Callable[[], set[str]]):
            返回当前允许授权的业务工具 key 集合的回调函数。

    返回值：
        Any: LangChain Tool 对象，请求名固定为 `request_tool_access`。

    异常说明：
        ValueError: 当申请的工具 key 不存在于允许集合中时由内部函数抛出。
    """

    @tool(
        description=(
                "申请当前任务所需的业务工具权限。"
                "当你需要调用未暴露的订单、商品、售后、用户或分析工具时，必须先调用本工具。"
                "参数必须传 tool_keys 数组和简短 reason。"
        ),
    )
    @tool_call_status(
        tool_name="申请工具权限",
        start_message="正在申请业务工具权限",
        error_message="申请业务工具权限失败",
        timely_message="业务工具权限正在持续处理中",
    )
    def request_tool_access(
            tool_keys: list[str],
            reason: str,
            runtime: ToolRuntime[None, AgentState],
    ) -> Command:
        """
        功能描述：
            为当前 agent 运行申请额外业务工具，并把授权结果写入状态。

        参数说明：
            tool_keys (list[str]): 需要授权的业务工具 key 数组。
            reason (str): 申请原因。
            runtime (ToolRuntime[None, AgentState]): 当前工具运行时上下文。

        返回值：
            Command:
                更新 `granted_tool_keys` 与一条 ToolMessage。
                当前实现已允许全部业务工具直接使用，因此该工具主要用于记录申请动作。

        异常说明：
            ValueError:
                - 当 `tool_keys/reason` 入参结构非法时由模型校验抛出；
                - 当状态中的 `granted_tool_keys` 结构非法时不会抛错，仅按空数组处理。
        """

        validated_request = RequestToolAccessRequest.model_validate(
            {
                "tool_keys": tool_keys,
                "reason": reason,
            }
        )
        normalized_tool_keys = validated_request.tool_keys
        normalized_reason = validated_request.reason
        allowed_tool_keys = get_allowed_tool_keys()
        unresolved_tool_keys = [
            tool_key
            for tool_key in normalized_tool_keys
            if tool_key not in allowed_tool_keys
        ]

        current_state = runtime.state if isinstance(runtime.state, Mapping) else {}
        current_granted_tool_keys = current_state.get("granted_tool_keys", [])
        if not isinstance(current_granted_tool_keys, list):
            current_granted_tool_keys = []

        merged_tool_keys = merge_unique_tool_keys(
            existing_tool_keys=[
                str(item).strip()
                for item in current_granted_tool_keys
                if str(item).strip()
            ],
            requested_tool_keys=sorted(allowed_tool_keys),
        )

        tool_message_lines = [
            "当前已允许全部业务工具直接使用，无需额外授权。",
            f"本次申请原因：{normalized_reason}",
        ]
        if unresolved_tool_keys:
            tool_message_lines.append(
                "以下名称不是实际注册工具名，已忽略："
                + ", ".join(unresolved_tool_keys)
            )
        tool_message_lines.append("请直接调用实际工具名继续完成任务。")
        tool_message = ToolMessage(
            content="\n".join(tool_message_lines),
            tool_call_id=runtime.tool_call_id,
        )
        return Command(
            update={
                "messages": [tool_message],
                "granted_tool_keys": merged_tool_keys,
            }
        )

    return request_tool_access


__all__ = [
    "KnowledgeSearchToolRequest",
    "RequestToolAccessRequest",
    "UserInfo",
    "create_request_tool_access_tool",
    "format_ids_to_string",
    "get_safe_user_info",
    "merge_unique_tool_keys",
    "normalize_id_list",
    "search_knowledge_context",
]
