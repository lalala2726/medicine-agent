from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage

from app.utils.prompt_utils import load_prompt


class BasePromptMiddleware(AgentMiddleware):
    """基础提示词中间件。

    作用：
        在模型调用前自动注入通用基础提示词，避免各节点手工拼接。

    设计目标：
        1. 幂等：同一次请求重复经过中间件时不重复追加；
        2. 与技能提示词兼容：若已存在 `## 技能系统` 段落，基础提示词插入到其前方；
        3. 不改动状态与工具，仅处理 `system_message`。
    """

    def __init__(
            self,
            *,
            base_prompt_file: str = "assistant/base_prompt.md",
            section_marker: str = "## 基础系统规则",
            skills_section_marker: str = "## 技能系统",
    ) -> None:
        """初始化基础提示词中间件。

        参数：
            base_prompt_file: 基础提示词文件相对路径（位于 `resources/prompt` 下）。
            section_marker: 基础提示词段落标识，用于防重复注入。
            skills_section_marker: 技能系统段落标识，用于控制插入顺序。
        """

        self.base_prompt_file = base_prompt_file
        self.section_marker = section_marker
        self.skills_section_marker = skills_section_marker
        self._base_prompt_text = load_prompt(base_prompt_file).strip()

    def _build_base_section(self) -> str:
        """构建基础提示词段落文本。"""

        if not self._base_prompt_text:
            return self.section_marker
        return f"{self.section_marker}\n\n{self._base_prompt_text}"

    def _inject_base_prompt(self, request: ModelRequest) -> ModelRequest:
        """向系统消息注入基础提示词。"""

        system_message = request.system_message
        current_text = system_message.text if system_message is not None else ""
        if self.section_marker in current_text:
            return request

        base_section = self._build_base_section()
        if not current_text:
            merged_text = base_section
        else:
            skills_index = current_text.find(self.skills_section_marker)
            if skills_index >= 0:
                before_skills = current_text[:skills_index].rstrip()
                skills_and_after = current_text[skills_index:].lstrip()
                if before_skills:
                    merged_text = f"{before_skills}\n\n{base_section}\n\n{skills_and_after}"
                else:
                    merged_text = f"{base_section}\n\n{skills_and_after}"
            else:
                merged_text = f"{current_text.rstrip()}\n\n{base_section}"

        return request.override(system_message=SystemMessage(content=merged_text))

    def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """同步包装模型调用，注入基础提示词后继续执行。"""

        modified_request = self._inject_base_prompt(request)
        return handler(modified_request)

    async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """异步包装模型调用，注入基础提示词后继续执行。"""

        modified_request = self._inject_base_prompt(request)
        return await handler(modified_request)
