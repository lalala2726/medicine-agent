from __future__ import annotations

from app.utils.resource_text_utils import (
    RESOURCES_DIR,
    load_resource_text,
    load_resource_text_from_root,
)

PROMPT_DIR = RESOURCES_DIR / "prompt"
_DEFAULT_PROMPT_DIR = PROMPT_DIR.resolve()
_PROMPT_CACHE: dict[str, str] = {}


class PromptUtils:
    """Prompt 读取工具。"""

    @staticmethod
    def load_prompt(name: str) -> str:
        """按相对路径读取提示词文件。

        作用：
            从 `resources/prompt` 目录安全读取 markdown 提示词，并做内存缓存。

        参数：
            name: 提示词相对路径，必须包含 `.md` 后缀。

        返回：
            str: 提示词文本内容。
        """

        resolved_prompt_dir = PROMPT_DIR.resolve()
        if resolved_prompt_dir == _DEFAULT_PROMPT_DIR:
            return load_resource_text(
                "prompt",
                name,
                allowed_suffixes=(".md",),
                cache=_PROMPT_CACHE,
            )

        # 兼容单测里 monkeypatch PROMPT_DIR 的场景。
        return load_resource_text_from_root(
            resolved_prompt_dir,
            name,
            allowed_suffixes=(".md",),
            cache=_PROMPT_CACHE,
        )


def load_prompt(name: str) -> str:
    """按相对路径读取 `resources/prompt` 下的 markdown 提示词。"""

    return PromptUtils.load_prompt(name)
