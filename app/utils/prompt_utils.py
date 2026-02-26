from __future__ import annotations

from pathlib import Path, PurePosixPath

PROMPT_DIR = Path(__file__).resolve().parents[2] / "resources" / "prompt"
_PROMPT_CACHE: dict[str, str] = {}


def _normalize_prompt_name(name: str) -> str:
    """规范化并校验提示词文件相对路径。

    作用：
        将调用方传入的名称统一为 `resources/prompt` 下的相对路径，
        并执行安全校验：
        1. 必须为非空字符串；
        2. 只能是相对路径；
        3. 不能包含 `..` 路径穿越；
        4. 仅允许 `.md` 后缀。

    参数：
        name: 提示词名称或相对路径，例如 `_system/base_prompt.md`、
            `system/order.md`。

    返回：
        str: 规范化后的相对路径（POSIX 分隔）。
    """

    raw_name = str(name or "").strip()
    if not raw_name:
        raise ValueError("Prompt name cannot be empty")

    normalized_name = raw_name.replace("\\", "/")
    posix_path = PurePosixPath(normalized_name)
    if posix_path.is_absolute():
        raise ValueError("Prompt path must be relative to resources/prompt")

    normalized_parts: list[str] = []
    for part in posix_path.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            raise ValueError("Prompt path cannot contain parent traversal '..'")
        normalized_parts.append(part)

    if not normalized_parts:
        raise ValueError("Prompt name cannot be empty")

    normalized_relative_path = "/".join(normalized_parts)
    if not normalized_relative_path.endswith(".md"):
        raise ValueError("Prompt path must include '.md' suffix")

    return normalized_relative_path


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

        normalized_name = _normalize_prompt_name(name)
        cached = _PROMPT_CACHE.get(normalized_name)
        if cached is not None:
            return cached

        prompt_root = PROMPT_DIR.resolve()
        file_path = (prompt_root / Path(*normalized_name.split("/"))).resolve()
        if not file_path.is_relative_to(prompt_root):
            raise ValueError("Prompt path escapes resources/prompt")
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"Prompt file not found: {file_path}")

        prompt_text = file_path.read_text(encoding="utf-8")
        _PROMPT_CACHE[normalized_name] = prompt_text
        return prompt_text


def load_prompt(name: str) -> str:
    """按相对路径读取 `resources/prompt` 下的 markdown 提示词。"""

    return PromptUtils.load_prompt(name)
