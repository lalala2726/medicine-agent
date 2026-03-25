"""
客户端 commerce 基础工具与共享辅助函数。

说明：
1. 基础工具默认直接暴露给 `commerce_agent`；
2. 订单、商品、售后业务工具通过动态加载机制按需暴露；
3. 该模块只负责工具目录与加载逻辑，不承载具体业务查询实现。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agent.client.state import AgentState
from app.core.agent.agent_tool_events import tool_call_status


def merge_unique_loaded_tool_keys(
        existing_tool_keys: list[str],
        requested_tool_keys: list[str],
) -> list[str]:
    """
    功能描述：
        合并已加载工具与本次新加载工具，并保持顺序去重。

    参数说明：
        existing_tool_keys (list[str]): 当前状态中已有的工具 key 数组。
        requested_tool_keys (list[str]): 本次新加载的工具 key 数组。

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


class LoadToolsRequest(BaseModel):
    """
    功能描述：
        commerce 工具加载入参模型。

    参数说明：
        tool_keys (list[str]): 本次需要加载的业务工具 key 数组。
        reason (str): 加载原因。

    返回值：
        无（数据模型定义）。

    异常说明：
        ValueError: 当工具数组或原因不合法时抛出。
    """

    model_config = ConfigDict(extra="forbid")

    tool_keys: list[str] = Field(
        min_length=1,
        description="需要加载的业务工具 key 数组，只允许 snake_case 工具名",
    )
    reason: str = Field(
        ...,
        min_length=1,
        description="本次加载工具的业务原因说明",
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
            规范化工具加载原因文本。

        参数说明：
            value (str): 原始加载原因。

        返回值：
            str: 去除首尾空白后的原因文本。

        异常说明：
            ValueError: 当原因为空时抛出。
        """

        normalized = value.strip()
        if not normalized:
            raise ValueError("reason 不能为空")
        return normalized


class LoadableToolsCatalog(BaseModel):
    """
    功能描述：
        可加载业务工具目录模型。

    参数说明：
        exact_tool_names (list[str]): 当前可加载业务工具的精确工具名数组。
        tools_by_domain (dict[str, list[str]]): 按领域分组后的工具名映射。
        supports_multi_load (bool): 是否支持单次同时加载多个工具。
        usage_tip (str): 工具目录使用提示。

    返回值：
        无（数据模型定义）。

    异常说明：
        无。
    """

    exact_tool_names: list[str] = Field(
        ...,
        description="当前可加载业务工具的精确工具名数组",
    )
    tools_by_domain: dict[str, list[str]] = Field(
        ...,
        description="按领域分组后的工具名映射",
    )
    supports_multi_load: bool = Field(
        ...,
        description="是否支持单次同时加载多个工具",
    )
    usage_tip: str = Field(
        ...,
        description="调用加载工具时的使用提示",
    )


def create_list_loadable_tools_tool(
        *,
        get_tool_catalog: Callable[[], dict[str, list[str]]],
) -> Any:
    """
    功能描述：
        创建查看可加载业务工具目录的基础工具。

    参数说明：
        get_tool_catalog (Callable[[], dict[str, list[str]]]):
            返回按领域分组的业务工具目录回调函数。

    返回值：
        Any: LangChain Tool 对象，请求名固定为 `list_loadable_tools`。

    异常说明：
        无。
    """

    @tool(
        description=(
                "查看当前可加载的 commerce 业务工具精确名称目录。"
                "当你不确定工具名、需要确认多个工具该怎么写，或准备一次加载多个工具时，先调用本工具。"
        ),
    )
    @tool_call_status(
        tool_name="查看可加载工具目录",
        start_message="正在查看可加载工具目录",
        error_message="查看可加载工具目录失败",
        timely_message="可加载工具目录正在持续整理中",
    )
    def list_loadable_tools() -> LoadableToolsCatalog:
        """
        功能描述：
            返回当前允许加载的业务工具精确名称目录。

        参数说明：
            无。

        返回值：
            LoadableToolsCatalog: 可加载工具目录结构。

        异常说明：
            无。
        """

        tools_by_domain = get_tool_catalog()
        exact_tool_names: list[str] = []
        for domain_tool_names in tools_by_domain.values():
            for raw_tool_name in domain_tool_names:
                tool_name = str(raw_tool_name or "").strip()
                if not tool_name:
                    continue
                if tool_name in exact_tool_names:
                    continue
                exact_tool_names.append(tool_name)

        return LoadableToolsCatalog(
            exact_tool_names=exact_tool_names,
            tools_by_domain=tools_by_domain,
            supports_multi_load=True,
            usage_tip=(
                "调用 load_tools 时，tool_keys 必须使用 exact_tool_names 中的精确值；"
                "支持一次传入多个工具名同时加载。"
            ),
        )

    return list_loadable_tools


def create_load_tools_tool(
        *,
        get_allowed_tool_keys: Callable[[], set[str]],
) -> Any:
    """
    功能描述：
        创建运行时工具动态加载工具。

    参数说明：
        get_allowed_tool_keys (Callable[[], set[str]]):
            返回当前允许加载的业务工具 key 集合的回调函数。

    返回值：
        Any: LangChain Tool 对象，请求名固定为 `load_tools`。

    异常说明：
        ValueError: 当加载的工具 key 不存在于允许集合中时由内部函数抛出。
    """

    @tool(
        description=(
                "加载当前任务所需的 commerce 业务工具。"
                "当你需要调用当前不可见的订单、商品或售后工具时，必须先调用本工具。"
                "这是工具加载步骤，不需要等待用户确认。"
                "tool_keys 支持一次传入多个精确工具名同时加载。"
        ),
    )
    @tool_call_status(
        tool_name="加载商城业务工具",
        start_message="正在加载商城业务工具",
        error_message="加载商城业务工具失败",
        timely_message="商城业务工具正在持续加载中",
    )
    def load_tools(
            tool_keys: list[str],
            reason: str,
            runtime: ToolRuntime[None, AgentState],
    ) -> Command:
        """
        功能描述：
            为当前 commerce agent 运行加载额外业务工具，并把加载结果写入状态。

        参数说明：
            tool_keys (list[str]): 需要加载的业务工具 key 数组。
            reason (str): 加载原因。
            runtime (ToolRuntime[None, AgentState]): 当前工具运行时上下文。

        返回值：
            Command:
                更新 `loaded_tool_keys` 与一条 ToolMessage，
                使后续模型调用可以看到本次已加载的业务工具。

        异常说明：
            ValueError:
                - 当 `tool_keys/reason` 入参结构非法时由模型校验抛出；
                - 当加载了未注册的业务工具 key 时抛出；
                - 当状态中的 `loaded_tool_keys` 结构非法时不会抛错，仅按空数组处理。
        """

        validated_request = LoadToolsRequest.model_validate(
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
        if unresolved_tool_keys:
            raise ValueError(
                "不允许加载以下工具: " + ", ".join(unresolved_tool_keys)
            )

        current_state = runtime.state if isinstance(runtime.state, Mapping) else {}
        current_loaded_tool_keys = current_state.get("loaded_tool_keys", [])
        if not isinstance(current_loaded_tool_keys, list):
            current_loaded_tool_keys = []

        merged_tool_keys = merge_unique_loaded_tool_keys(
            existing_tool_keys=[
                str(item).strip()
                for item in current_loaded_tool_keys
                if str(item).strip()
            ],
            requested_tool_keys=normalized_tool_keys,
        )

        tool_message = ToolMessage(
            content="\n".join(
                [
                    "已加载以下 commerce 业务工具，可继续直接调用："
                    + ", ".join(normalized_tool_keys),
                    f"本次加载原因：{normalized_reason}",
                    "这些工具无需用户确认；你可以继续直接调用已加载的实际工具名完成任务。",
                ]
            ),
            tool_call_id=runtime.tool_call_id,
        )
        return Command(
            update={
                "messages": [tool_message],
                "loaded_tool_keys": merged_tool_keys,
            }
        )

    return load_tools


__all__ = [
    "LoadToolsRequest",
    "LoadableToolsCatalog",
    "create_list_loadable_tools_tool",
    "create_load_tools_tool",
    "merge_unique_loaded_tool_keys",
]
