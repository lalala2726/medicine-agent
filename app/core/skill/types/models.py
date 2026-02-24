from __future__ import annotations

from pathlib import Path
from typing import NotRequired, TypeAlias, TypedDict


class SkillExtraMetadata(TypedDict):
    """技能扩展元数据。

    作用：
        约束 `metadata` 字段结构，避免使用无约束字典。

    字段：
        author: 技能作者或维护者。
        version: 技能版本号。
    """

    author: NotRequired[str]
    version: NotRequired[str]


class SkillMetadata(TypedDict):
    """技能元数据。

    作用：
        表示在预加载阶段暴露给模型的最小技能信息。

    字段：
        name: 技能唯一标识，供 `load_skill` 调用使用。
        description: 技能描述信息，用于系统提示词展示。
        license: 技能许可协议（可选）。
        metadata: 扩展元数据（可选）。
    """

    name: str
    description: str
    license: NotRequired[str]
    metadata: NotRequired[SkillExtraMetadata]


SkillFileIndex: TypeAlias = dict[str, Path]
