from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
import yaml

from app.core.skill.discovery.scope import _is_path_within_root, normalize_scope
from app.core.skill.types.models import SkillExtraMetadata, SkillFileIndex, SkillMetadata


def _parse_frontmatter(raw_content: str) -> dict[str, Any]:
    """解析 frontmatter 文本块。

    作用：
        使用 YAML 解析 `--- ... ---` 区块，提取技能元数据。

    参数：
        raw_content: frontmatter 原始文本（通常只包含头部块）。

    返回：
        dict[str, Any]: 解析得到的键值对；若格式不合法则返回空字典。
    """

    lines = raw_content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}

    yaml_lines: list[str] = []
    for line in lines[1:]:
        if line.strip() == "---":
            break
        yaml_lines.append(line)

    yaml_text = "\n".join(yaml_lines).strip()
    if not yaml_text:
        return {}

    try:
        parsed = yaml.safe_load(yaml_text)
    except yaml.YAMLError:
        return {}

    if not isinstance(parsed, dict):
        return {}
    return parsed


def _parse_skill_extra_metadata(raw_metadata: Any, skill_file: Path) -> SkillExtraMetadata | None:
    """解析技能扩展元数据。

    作用：
        将 frontmatter 中的 `metadata` 转为强类型结构，仅保留 `author/version`。

    参数：
        raw_metadata: frontmatter 中的 `metadata` 原始值。
        skill_file: 当前技能文件路径，用于日志输出。

    返回：
        SkillExtraMetadata | None:
            - 合法且至少包含一个字段时返回对象；
            - 不存在、类型错误或无有效字段时返回 `None`。
    """

    if raw_metadata is None:
        return None
    if not isinstance(raw_metadata, dict):
        logger.warning(
            "技能 metadata 字段类型无效（应为对象），已忽略: {}",
            skill_file,
        )
        return None

    parsed: SkillExtraMetadata = {}
    author = str(raw_metadata.get("author") or "").strip()
    if author:
        parsed["author"] = author

    version = str(raw_metadata.get("version") or "").strip()
    if version:
        parsed["version"] = version

    if not parsed:
        logger.warning(
            "技能 metadata 中未包含有效字段（author/version），已忽略: {}",
            skill_file,
        )
        return None
    return parsed


def _read_frontmatter_block(skill_file: Path) -> str:
    """仅读取 `SKILL.md` 的 frontmatter 区块。

    作用：
        只读取文件头部 `--- ... ---` 内容，避免在预加载阶段读取全文。

    参数：
        skill_file: 技能文件路径。

    返回：
        str: frontmatter 原文；无 frontmatter 或读取失败时返回空字符串。
    """

    try:
        with skill_file.open("r", encoding="utf-8") as file_obj:
            first_line = file_obj.readline()
            if first_line.strip() != "---":
                return ""

            chunks = [first_line]
            for line in file_obj:
                chunks.append(line)
                if line.strip() == "---":
                    break
            return "".join(chunks)
    except OSError:
        return ""


def discover_skills_metadata(scope: str | None) -> tuple[list[SkillMetadata], SkillFileIndex]:
    """发现指定作用域下的技能元数据，并建立内部索引。

    作用：
        按固定规则扫描 `<scope>/*/SKILL.md`（非递归），解析 frontmatter，
        返回对模型可见的元数据列表，以及仅内部使用的 `name -> 文件路径` 索引。

    参数：
        scope: 技能作用域。为空时表示扫描技能根目录。

    返回：
        tuple[list[SkillMetadata], SkillFileIndex]:
            - skills_metadata: 对模型暴露的技能元数据（必填字段 + 可选字段）
            - skill_file_index: `frontmatter.name -> SKILL.md` 路径映射
    """

    _, scope_dir = normalize_scope(scope)
    if not scope_dir.exists() or not scope_dir.is_dir():
        return [], {}

    metadata_by_name: dict[str, SkillMetadata] = {}
    skill_file_index: SkillFileIndex = {}
    for child_dir in sorted(scope_dir.iterdir(), key=lambda path: path.name):
        if not child_dir.is_dir():
            continue
        if not _is_path_within_root(child_dir):
            continue

        skill_file = child_dir / "SKILL.md"
        if not skill_file.is_file():
            continue
        if not _is_path_within_root(skill_file):
            continue

        frontmatter_text = _read_frontmatter_block(skill_file)
        parsed = _parse_frontmatter(frontmatter_text)
        skill_name = str(parsed.get("name") or "").strip()
        if not skill_name:
            logger.error("跳过技能，frontmatter 缺少必填字段 name: {}", skill_file)
            continue

        description = str(parsed.get("description") or "").strip()
        if not description:
            logger.error("跳过技能，frontmatter 缺少必填字段 description: {}", skill_file)
            continue

        skill_metadata: SkillMetadata = {
            "name": skill_name,
            "description": description,
        }
        license_name = str(parsed.get("license") or "").strip()
        if license_name:
            skill_metadata["license"] = license_name

        extra_metadata = _parse_skill_extra_metadata(parsed.get("metadata"), skill_file)
        if extra_metadata is not None:
            skill_metadata["metadata"] = extra_metadata

        existing = skill_file_index.get(skill_name)
        if existing is not None:
            logger.warning(
                "Duplicate skill name '{}' in scope '{}', override {} -> {}",
                skill_name,
                scope,
                existing,
                skill_file,
            )

        skill_file_index[skill_name] = skill_file
        metadata_by_name[skill_name] = skill_metadata

    # 元数据按名称稳定排序，保证注入顺序可预期。
    deduped_metadata = [metadata_by_name[name] for name in sorted(skill_file_index.keys())]

    return deduped_metadata, skill_file_index


def discover_skills(scope: str | None = None) -> tuple[list[SkillMetadata], SkillFileIndex]:
    """对外暴露的技能发现入口。

    作用：
        统一封装技能元数据发现逻辑，供中间件或测试调用。

    参数：
        scope: 技能作用域。为空时表示扫描技能根目录。

    返回：
        tuple[list[SkillMetadata], SkillFileIndex]:
            - 技能元数据列表
            - 技能文件索引映射
    """

    return discover_skills_metadata(scope)
