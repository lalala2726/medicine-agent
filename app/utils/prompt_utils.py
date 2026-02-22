from __future__ import annotations

from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parents[2] / "resources" / "prompt"
_PROMPT_CACHE: dict[str, str] = {}


def _normalize_prompt_name(name: str) -> str:
    prompt_name = str(name or "").strip()
    if not prompt_name:
        raise ValueError("Prompt name cannot be empty")

    if prompt_name.endswith(".md"):
        prompt_name = prompt_name[:-3]
    if not prompt_name:
        raise ValueError("Prompt name cannot be empty")
    return prompt_name


class PromptUtils:
    """Prompt 读取工具。"""

    @staticmethod
    def load_prompt(name: str) -> str:
        normalized_name = _normalize_prompt_name(name)
        cached = _PROMPT_CACHE.get(normalized_name)
        if cached is not None:
            return cached

        file_path = PROMPT_DIR / f"{normalized_name}.md"
        if not file_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {file_path}")

        prompt_text = file_path.read_text(encoding="utf-8")
        _PROMPT_CACHE[normalized_name] = prompt_text
        return prompt_text


def load_prompt(name: str) -> str:
    """按名称读取 resources/prompt 下的 markdown 提示词。"""

    return PromptUtils.load_prompt(name)
