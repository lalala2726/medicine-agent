from __future__ import annotations

from langchain_core.prompts import SystemMessagePromptTemplate

from app.core.skill.types.models import SkillMetadata

SKILLS_SYSTEM_PROMPT = """
    ## 技能系统
    
    **可用技能：**
    
    {skills_list}

    **字段含义说明：**

    - `name`：技能唯一标识，调用 `load_skill("<name>")` 时必须使用该值。
    - `description`：技能用途与适用场景描述，用于判断是否命中该技能。
    - `license`：技能许可信息（可选）。
    - `metadata.author`：技能作者/维护者（可选）。
    - `metadata.version`：技能版本（可选）。
    
    **如何使用技能（渐进式披露）：**
    
    技能采用“渐进式披露”模式——你可以在上方看到技能名称和描述，但只有在需要时才读取完整说明：
    
    1. **识别何时适用某个技能**：检查用户任务是否匹配某个技能的描述
    2. **按名称加载该技能的完整说明**：调用 `load_skill("<skill_name>")`
    
    **何时使用技能：**
    
    - 用户请求匹配某个技能的领域（例如：“最新的运营情况” → 使用 analysis 技能）
    - 你需要专业知识或结构化工作流程
    - 某个技能为复杂任务提供经过验证的模式
    
    **示例工作流程：**
    
    用户：“你能帮我分析一下最新的运营情况吗？”
    
    1. 检查可用技能 → 看到 “analysis” 技能
    2. 调用 `load_skill("analysis")` 读取完整技能说明
    3. 按照技能中的流程和约束执行
    
    请记住：技能可以让你更强大、更一致。当不确定时，检查是否存在适用于该任务的技能！
""".strip()


def _format_skills_list(skills_metadata: list[SkillMetadata]) -> str:
    """将技能元数据渲染为提示词列表文本。

    作用：
        将技能元数据按“原始字段”形式渲染为 YAML 风格块，直接提供给模型。

    参数：
        skills_metadata: 技能元数据列表。

    返回：
        str: 可直接插入模板的技能列表文本。
    """

    if not skills_metadata:
        return "- （暂无可用技能）"

    items: list[str] = []
    for item in skills_metadata:
        lines = [
            "```yaml",
            f"name: {item['name']}",
            f"description: {item.get('description') or ''}",
        ]

        license_name = str(item.get("license") or "").strip()
        if license_name:
            lines.append(f"license: {license_name}")

        extra_metadata = item.get("metadata")
        if isinstance(extra_metadata, dict):
            author = str(extra_metadata.get("author") or "").strip()
            version = str(extra_metadata.get("version") or "").strip()
            if author or version:
                lines.append("metadata:")
                if author:
                    lines.append(f"  author: {author}")
                if version:
                    lines.append(f"  version: {version}")

        lines.append("```")
        items.append("\n".join(lines))

    return "\n\n".join(items)


def build_skills_prompt(
    skills_metadata: list[SkillMetadata],
    *,
    system_prompt_template: str = SKILLS_SYSTEM_PROMPT,
) -> str:
    """基于模板构建技能系统提示词段落。

    作用：
        使用 `SystemMessagePromptTemplate` 将 `{skills_list}` 占位符替换为
        当前可用技能列表，生成最终注入模型的文本。

    参数：
        skills_metadata: 技能元数据列表。
        system_prompt_template: 技能系统提示词模板，默认使用内置模板。

    返回：
        str: 渲染后的技能提示词文本。
    """

    template = SystemMessagePromptTemplate.from_template(system_prompt_template)
    formatted = template.format(skills_list=_format_skills_list(skills_metadata))
    return formatted.text
