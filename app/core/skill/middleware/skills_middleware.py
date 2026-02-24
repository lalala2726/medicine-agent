from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, NotRequired, TypedDict

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage

from app.core.skill.discovery.metadata import _discover_skills_metadata
from app.core.skill.discovery.scope import _normalize_scope
from app.core.skill.prompt.templates import SKILLS_SYSTEM_PROMPT, build_skills_prompt
from app.core.skill.tool.load_skill import create_load_skill_tool
from app.core.skill.types.models import SkillExtraMetadata, SkillFileIndex, SkillMetadata


class SkillMiddlewareState(TypedDict, total=False):
    """技能中间件状态结构。

    作用：
        声明运行时需要保留的中间件状态键，避免被 LangChain 状态系统丢弃。

    字段：
        skills_metadata: 预加载后的技能元数据列表。
    """

    skills_metadata: NotRequired[list[SkillMetadata]]


class SkillMiddleware(AgentMiddleware):
    """技能中间件。

    作用：
        1. 在 `before_agent/abefore_agent` 阶段预加载技能元数据；
        2. 在模型调用前向系统提示词注入技能列表与使用说明；
        3. 注册 `load_skill` 工具，支持按名称懒加载完整技能文件。
    """

    state_schema = SkillMiddlewareState

    def __init__(self, scope: str, system_prompt_template: str = SKILLS_SYSTEM_PROMPT):
        """初始化技能中间件。

        参数：
            scope: 技能作用域，例如 `supervisor`、`supervisor/a`。
            system_prompt_template: 技能提示词模板，默认使用内置模板。

        返回：
            None
        """

        normalized_scope, _ = _normalize_scope(scope)
        self.scope = normalized_scope
        self._scope_marker = "## 技能系统"
        self._system_prompt_template = system_prompt_template
        self._skill_file_index: SkillFileIndex = {}
        self._load_skill_tool = create_load_skill_tool(
            normalized_scope,
            get_skill_file_index=lambda: self._skill_file_index,
        )
        self.tools = [self._load_skill_tool]

    def _build_skills_section(self, skills_metadata: list[SkillMetadata]) -> str:
        """构建技能提示词段落。

        参数：
            skills_metadata: 技能元数据列表（必填字段 + 可选字段）。

        返回：
            str: 渲染后的技能系统提示词文本。
        """

        return build_skills_prompt(
            skills_metadata,
            system_prompt_template=self._system_prompt_template,
        )

    def _inject_skills_prompt(self, request: ModelRequest) -> ModelRequest:
        """向请求的系统消息注入技能提示词。

        参数：
            request: 当前模型请求对象。

        返回：
            ModelRequest: 注入后（或原样）的模型请求对象。
        """

        raw_metadata = request.state.get("skills_metadata", [])
        skills_metadata: list[SkillMetadata] = []
        if isinstance(raw_metadata, list):
            for item in raw_metadata:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "").strip()
                if not name:
                    continue
                description = str(item.get("description") or "").strip()
                if not description:
                    continue

                skill_item: SkillMetadata = {"name": name, "description": description}
                license_name = str(item.get("license") or "").strip()
                if license_name:
                    skill_item["license"] = license_name
                extra_metadata = item.get("metadata")
                if isinstance(extra_metadata, dict):
                    parsed_metadata: SkillExtraMetadata = {}
                    author = str(extra_metadata.get("author") or "").strip()
                    if author:
                        parsed_metadata["author"] = author
                    version = str(extra_metadata.get("version") or "").strip()
                    if version:
                        parsed_metadata["version"] = version
                    if parsed_metadata:
                        skill_item["metadata"] = parsed_metadata

                skills_metadata.append(skill_item)

        skills_section = self._build_skills_section(skills_metadata)
        system_message = request.system_message
        if system_message is not None and self._scope_marker in system_message.text:
            return request

        system_text = ""
        if system_message is not None:
            system_text = system_message.text

        merged_text = (
            f"{system_text.rstrip()}\n\n{skills_section}" if system_text else skills_section
        )
        return request.override(system_message=SystemMessage(content=merged_text))

    def before_agent(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        """同步阶段预加载技能元数据并刷新内部索引。

        参数：
            state: 运行时状态字典。
            runtime: 运行时上下文（此处不直接使用）。

        返回：
            dict[str, Any] | None:
                - 当 `state` 中尚无 `skills_metadata` 时，返回增量状态；
                - 若已存在则返回 `None`（幂等）。
        """

        _ = runtime
        skills_metadata, skill_file_index = _discover_skills_metadata(self.scope)
        self._skill_file_index = skill_file_index
        if "skills_metadata" in state:
            return None
        return {"skills_metadata": skills_metadata}

    async def abefore_agent(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        """异步阶段预加载技能元数据。

        参数：
            state: 运行时状态字典。
            runtime: 运行时上下文。

        返回：
            dict[str, Any] | None: 与 `before_agent` 保持一致。
        """

        return self.before_agent(state, runtime)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ):
        """同步包装模型调用，注入技能提示词后继续执行。

        参数：
            request: 当前模型请求。
            handler: 下游模型调用处理器。

        返回：
            ModelResponse: 下游处理器返回的模型响应。
        """

        modified_request = self._inject_skills_prompt(request)
        return handler(modified_request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ):
        """异步包装模型调用，注入技能提示词后继续执行。

        参数：
            request: 当前模型请求。
            handler: 下游异步模型调用处理器。

        返回：
            ModelResponse: 下游处理器返回的模型响应。
        """

        modified_request = self._inject_skills_prompt(request)
        return await handler(modified_request)
