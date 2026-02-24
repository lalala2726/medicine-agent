from __future__ import annotations

from collections.abc import Callable

from langchain_core.tools import tool

from app.core.skill.discovery.scope import _normalize_scope
from app.core.skill.types.models import SkillFileIndex


def create_load_skill_tool(
    scope: str,
    get_skill_file_index: Callable[[], SkillFileIndex],
):
    """创建 `load_skill` 工具函数。

    作用：
        在固定 `scope` 下基于预构建索引提供技能全文懒加载能力。
        该函数返回一个可注册到 LangChain 的工具对象。

    参数：
        scope: 技能作用域，用于错误提示与作用域校验。
        get_skill_file_index: 获取最新技能索引的回调函数，返回 `name -> 文件路径`。

    返回：
        Callable: 可供模型调用的 `load_skill(skill_name)` 工具。
    """

    normalized_scope, _ = _normalize_scope(scope)

    @tool(
        description=(
            "按技能名称加载完整 SKILL.md 内容。"
            "当任务命中某个技能并需要详细说明时调用。"
        )
    )
    def load_skill(skill_name: str) -> str:
        """按技能名称读取并返回完整技能文件内容。

        作用：
            仅在模型显式调用时读取对应 `SKILL.md` 全文，实现渐进式加载。

        参数：
            skill_name: 技能名称，对应 frontmatter 中的 `name` 字段。

        返回：
            str: 命中时返回 `Loaded skill: ...` 与全文；未命中返回可用技能列表提示。
        """

        normalized_skill_name = str(skill_name or "").strip()
        if not normalized_skill_name:
            return "Skill name cannot be empty."

        skill_file_index = get_skill_file_index()
        selected_file = skill_file_index.get(normalized_skill_name)
        if selected_file is None:
            available_names = sorted(skill_file_index.keys())
            available_text = ", ".join(available_names) if available_names else "(none)"
            return (
                f"Skill '{normalized_skill_name}' not found under scope '{normalized_scope}'. "
                f"Available skills: {available_text}"
            )

        try:
            full_content = selected_file.read_text(encoding="utf-8")
        except OSError:
            return f"Failed to read skill file for '{normalized_skill_name}'."
        return f"Loaded skill: {normalized_skill_name}\n\n{full_content}"

    return load_skill
