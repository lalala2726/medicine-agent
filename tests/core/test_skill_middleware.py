from __future__ import annotations

from pathlib import Path
import importlib
from typing import Any

import pytest
from langchain.agents import create_agent as lc_create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage, SystemMessage
from loguru import logger

import app.core.skill.discovery.scope as scope_module
from app.core.skill import SkillMiddleware, create_load_skill_tool, discover_skills


class _ToolFriendlyFakeListChatModel(FakeListChatModel):
    """支持工具绑定的测试模型。"""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):  # type: ignore[no-untyped-def]
        _ = (tools, tool_choice, kwargs)
        return self


def _write_skill(
    root: Path,
    relative_dir: str,
    description: str = "demo desc",
    *,
    skill_name: str = "Demo Skill",
    license_name: str | None = None,
    metadata: dict[str, Any] | None = None,
    include_name: bool = True,
    include_description: bool = True,
) -> None:
    """构造测试用 `SKILL.md` 文件。

    测试目的：
        统一生成不同 frontmatter 组合，降低各用例样板代码。

    预期结果：
        在指定目录下写入可被 discovery/load_skill 识别的技能文件。
    """

    skill_file = root / relative_dir / "SKILL.md"
    skill_file.parent.mkdir(parents=True, exist_ok=True)
    frontmatter_lines = ["---"]
    if include_name:
        frontmatter_lines.append(f"name: {skill_name}")
    if include_description:
        frontmatter_lines.append(f"description: {description}")
    if license_name is not None:
        frontmatter_lines.append(f"license: {license_name}")
    if metadata is not None:
        frontmatter_lines.append("metadata:")
        for key, value in metadata.items():
            frontmatter_lines.append(f'  {key}: "{value}"')
    frontmatter_lines.append("---")

    content = "\n".join(frontmatter_lines) + "\n\n# Steps\n- step 1\n"
    skill_file.write_text(content, encoding="utf-8")


def test_discover_skills_only_reads_immediate_children_for_scope_supervisor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证 scope=supervisor 只扫描直系子目录。

    测试目的：
        防止递归扫描把更深层级技能误加载。

    预期结果：
        仅返回 `supervisor/*/SKILL.md` 中的 a/b/c。
    """

    skills_root = tmp_path / "skills"
    _write_skill(skills_root, "supervisor/a", "desc a", skill_name="a")
    _write_skill(skills_root, "supervisor/b", "desc b", skill_name="b")
    _write_skill(skills_root, "supervisor/c", "desc c", skill_name="c")
    _write_skill(skills_root, "supervisor/c/a", "nested should be ignored", skill_name="c-a")
    _write_skill(skills_root, "supervisor/d/w", "deep should be ignored", skill_name="d-w")
    monkeypatch.setattr(scope_module, "SKILLS_ROOT", skills_root)

    metadata, _ = discover_skills("supervisor")

    assert [item["name"] for item in metadata] == ["a", "b", "c"]


def test_discover_skills_only_reads_immediate_children_for_nested_scope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证嵌套 scope 仍只扫描当前层级的直系子目录。

    测试目的：
        保证 `scope=supervisor/a` 时只读取 `supervisor/a/*/SKILL.md`。

    预期结果：
        返回 b/c，忽略 `supervisor/a/c/s` 等更深目录。
    """

    skills_root = tmp_path / "skills"
    _write_skill(skills_root, "supervisor/a/b", "desc b", skill_name="b")
    _write_skill(skills_root, "supervisor/a/c", "desc c", skill_name="c")
    _write_skill(skills_root, "supervisor/a/c/s", "nested should be ignored", skill_name="c-s")
    monkeypatch.setattr(scope_module, "SKILLS_ROOT", skills_root)

    metadata, _ = discover_skills("supervisor/a")

    assert [item["name"] for item in metadata] == ["b", "c"]


def test_discover_skills_raises_when_scope_depth_exceeds_three_levels() -> None:
    """验证 scope 层级超限时会被拒绝。

    测试目的：
        确认最大 3 级作用域限制生效。

    预期结果：
        传入四级 scope 时抛出 `ValueError`。
    """

    with pytest.raises(ValueError):
        discover_skills("supervisor/a/c/d")


def test_before_agent_is_idempotent_when_skills_metadata_already_exists() -> None:
    """验证 `before_agent` 的幂等行为。

    测试目的：
        避免已存在 `skills_metadata` 时重复覆盖状态。

    预期结果：
        `before_agent` 返回 `None`。
    """

    middleware = SkillMiddleware(scope="supervisor")
    state = {"skills_metadata": [{"name": "existing", "description": "cached"}]}

    result = middleware.before_agent(state, runtime=None)

    assert result is None


def test_before_agent_returns_metadata_without_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证预加载只返回元数据，不暴露路径/正文。

    测试目的：
        确认 `before_agent` 仅注入前置元数据，并携带可选字段。

    预期结果：
        返回 `name/description/license/metadata`，且不含 `path/content`。
    """

    skills_root = tmp_path / "skills"
    _write_skill(
        skills_root,
        "supervisor/analysis",
        "analysis desc",
        skill_name="analysis",
        license_name="Apache-2.0",
        metadata={"author": "example-org", "version": "1.0"},
    )
    monkeypatch.setattr(scope_module, "SKILLS_ROOT", skills_root)
    middleware = SkillMiddleware(scope="supervisor")

    result = middleware.before_agent({}, runtime=None)

    assert result is not None
    metadata = result["skills_metadata"]
    assert metadata == [
        {
            "name": "analysis",
            "description": "analysis desc",
            "license": "Apache-2.0",
            "metadata": {"author": "example-org", "version": "1.0"},
        }
    ]
    assert "path" not in metadata[0]
    assert "content" not in metadata[0]


def test_discover_skills_only_keeps_author_and_version_in_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证 metadata 仅保留强约束字段。

    测试目的：
        确保 `metadata` 只透传 `author/version`，忽略其他键。

    预期结果：
        `channel` 被过滤，仅剩 `author/version`。
    """

    skills_root = tmp_path / "skills"
    _write_skill(
        skills_root,
        "supervisor/analysis",
        "analysis desc",
        skill_name="analysis",
        metadata={"author": "example-org", "version": "1.0", "channel": "dev"},
    )
    monkeypatch.setattr(scope_module, "SKILLS_ROOT", skills_root)

    metadata, _ = discover_skills("supervisor")

    assert metadata == [
        {
            "name": "analysis",
            "description": "analysis desc",
            "metadata": {"author": "example-org", "version": "1.0"},
        }
    ]


def test_discover_skills_skips_missing_required_fields_and_logs_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证缺少必填字段时跳过并记录错误日志。

    测试目的：
        确认 `name/description` 必填校验与错误可观测性。

    预期结果：
        相关技能不被加载，日志中包含错误级别与具体文件路径。
    """

    skills_root = tmp_path / "skills"
    _write_skill(
        skills_root,
        "supervisor/missing_name",
        "desc",
        include_name=False,
    )
    _write_skill(
        skills_root,
        "supervisor/missing_description",
        include_description=False,
    )
    monkeypatch.setattr(scope_module, "SKILLS_ROOT", skills_root)

    logs: list[str] = []
    log_id = logger.add(logs.append, format="{level}|{message}")
    try:
        metadata, _ = discover_skills("supervisor")
    finally:
        logger.remove(log_id)

    assert metadata == []
    all_logs = "\n".join(logs)
    assert "ERROR|跳过技能，frontmatter 缺少必填字段 name" in all_logs
    assert "ERROR|跳过技能，frontmatter 缺少必填字段 description" in all_logs
    assert "missing_name/SKILL.md" in all_logs
    assert "missing_description/SKILL.md" in all_logs


def test_load_skill_returns_full_skill_content(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证 `load_skill` 可按名称返回全文。

    测试目的：
        确认技能正文采取按需懒加载策略且内容完整。

    预期结果：
        返回结果包含 `Loaded skill` 标记、frontmatter 与正文片段。
    """

    skills_root = tmp_path / "skills"
    _write_skill(skills_root, "supervisor/analysis", "analysis desc", skill_name="analysis")
    monkeypatch.setattr(scope_module, "SKILLS_ROOT", skills_root)

    _, skill_file_index = discover_skills("supervisor")
    load_skill = create_load_skill_tool("supervisor", lambda: skill_file_index)
    content = load_skill.invoke({"skill_name": "analysis"})

    assert "Loaded skill: analysis" in content
    assert "description: analysis desc" in content
    assert "# Steps" in content


def test_load_skill_returns_available_names_when_target_not_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证 `load_skill` 未命中时的提示信息。

    测试目的：
        确保调用错误时能给出可用技能名，便于模型重试。

    预期结果：
        返回 `not found`，并列出可用技能 `analysis`。
    """

    skills_root = tmp_path / "skills"
    _write_skill(skills_root, "supervisor/analysis", "analysis desc", skill_name="analysis")
    monkeypatch.setattr(scope_module, "SKILLS_ROOT", skills_root)

    _, skill_file_index = discover_skills("supervisor")
    load_skill = create_load_skill_tool("supervisor", lambda: skill_file_index)
    content = load_skill.invoke({"skill_name": "missing"})

    assert "not found" in content.lower()
    assert "analysis" in content


def test_load_skill_uses_frontmatter_name_not_directory_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证 `load_skill` 以 frontmatter.name 作为唯一键。

    测试目的：
        防止目录名与技能名不一致时发生错误映射。

    预期结果：
        `analysis` 可加载，`analysis_dir` 未命中。
    """

    skills_root = tmp_path / "skills"
    _write_skill(
        skills_root,
        "supervisor/analysis_dir",
        "analysis desc",
        skill_name="analysis",
    )
    monkeypatch.setattr(scope_module, "SKILLS_ROOT", skills_root)

    _, skill_file_index = discover_skills("supervisor")
    load_skill = create_load_skill_tool("supervisor", lambda: skill_file_index)

    ok_content = load_skill.invoke({"skill_name": "analysis"})
    missing_content = load_skill.invoke({"skill_name": "analysis_dir"})

    assert "Loaded skill: analysis" in ok_content
    assert "not found" in missing_content.lower()


def test_before_agent_preloads_metadata_but_not_full_content(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证预加载不读取正文，工具调用时才读取。

    测试目的：
        保证渐进式加载：`before_agent` 只处理元数据。

    预期结果：
        `before_agent` 阶段读取次数为 0；调用 `load_skill` 后读取次数为 1。
    """

    skills_root = tmp_path / "skills"
    _write_skill(skills_root, "supervisor/analysis", "analysis desc", skill_name="analysis")
    monkeypatch.setattr(scope_module, "SKILLS_ROOT", skills_root)

    read_calls: list[Path] = []
    original_read_text = Path.read_text

    def _spy_read_text(path_obj: Path, *args, **kwargs):  # type: ignore[no-untyped-def]
        read_calls.append(path_obj)
        return original_read_text(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _spy_read_text)
    middleware = SkillMiddleware(scope="supervisor")
    middleware.before_agent({}, runtime=None)
    assert read_calls == []

    content = middleware.tools[0].invoke({"skill_name": "analysis"})
    assert "Loaded skill: analysis" in content
    assert len(read_calls) == 1


def test_wrap_model_call_injects_skills_system_prompt_without_paths() -> None:
    """验证系统提示词注入内容正确。

    测试目的：
        确认注入文本包含技能元数据与调用说明，但不泄露路径或全文。

    预期结果：
        包含 `name/description/license/metadata` 与 `load_skill("<skill_name>")`，
        不包含 `SKILL.md`、`resources/skills`。
    """

    middleware = SkillMiddleware(scope="supervisor")
    request = ModelRequest(
        model=object(),
        messages=[],
        system_message=SystemMessage(content="base prompt"),
        tools=[],
        state={
            "skills_metadata": [
                {
                    "name": "analysis",
                    "description": "analysis desc",
                    "license": "Apache-2.0",
                    "metadata": {"author": "example-org", "version": "1.0"},
                }
            ]
        },
        runtime=None,
    )

    captured_request: dict[str, ModelRequest] = {}

    def handler(modified_request: ModelRequest) -> ModelResponse:
        captured_request["request"] = modified_request
        return ModelResponse(result=[AIMessage(content="ok")])

    middleware.wrap_model_call(request, handler)
    system_text = captured_request["request"].system_message.text

    assert "## 技能系统" in system_text
    assert "如何使用技能（渐进式披露）" in system_text
    assert "字段含义说明" in system_text
    assert "name: analysis" in system_text
    assert "description: analysis desc" in system_text
    assert "license: Apache-2.0" in system_text
    assert "metadata:" in system_text
    assert "author: example-org" in system_text
    assert "version: 1.0" in system_text
    assert "load_skill(\"<name>\")" in system_text
    assert "load_skill(\"<skill_name>\")" in system_text
    assert "SKILL.md" not in system_text
    assert "resources/skills" not in system_text


def test_runtime_state_preserves_skills_metadata_for_prompt_injection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证真实运行链路中 `skills_metadata` 不会被状态系统丢弃。

    测试目的：
        覆盖 `before_agent -> wrap_model_call` 的实际集成路径，确保可用技能可注入提示词。

    预期结果：
        `_build_skills_section` 收到非空技能元数据，且包含预加载的技能名称。
    """

    skills_root = tmp_path / "skills"
    _write_skill(skills_root, "supervisor/analysis", "analysis desc", skill_name="analysis")
    monkeypatch.setattr(scope_module, "SKILLS_ROOT", skills_root)

    middleware = SkillMiddleware(scope="supervisor")
    captured_metadata: list[list[dict[str, Any]]] = []
    original_build = middleware._build_skills_section

    def _spy_build(skills_metadata):  # type: ignore[no-untyped-def]
        captured_metadata.append(list(skills_metadata))
        return original_build(skills_metadata)

    monkeypatch.setattr(middleware, "_build_skills_section", _spy_build)

    agent = lc_create_agent(
        model=_ToolFriendlyFakeListChatModel(responses=["ok"]),
        middleware=[middleware],
    )
    agent.invoke({"messages": [{"role": "user", "content": "hi"}]})

    assert captured_metadata
    assert captured_metadata[0]
    assert captured_metadata[0][0]["name"] == "analysis"


def test_old_skill_module_path_removed() -> None:
    """验证旧模块入口已移除。

    测试目的：
        防止调用方继续依赖已废弃的旧导入路径。

    预期结果：
        导入 `app.core.agent.skill` 抛出 `ModuleNotFoundError`。
    """

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("app.core.agent.skill")
