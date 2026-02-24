from __future__ import annotations

import pytest

import app.utils.prompt_utils as prompt_utils_module
from app.utils.prompt_utils import load_prompt


@pytest.fixture(autouse=True)
def reset_prompt_cache(monkeypatch):
    monkeypatch.setattr(prompt_utils_module, "_PROMPT_CACHE", {})


def test_load_prompt_reads_markdown_text(monkeypatch, tmp_path):
    prompt_file = tmp_path / "assistant_demo_prompt.md"
    prompt_file.write_text("hello prompt", encoding="utf-8")
    monkeypatch.setattr(prompt_utils_module, "PROMPT_DIR", tmp_path)

    assert load_prompt("assistant_demo_prompt.md") == "hello prompt"


def test_load_prompt_trims_name_whitespace(monkeypatch, tmp_path):
    prompt_file = tmp_path / "assistant_trim_prompt.md"
    prompt_file.write_text("trim ok", encoding="utf-8")
    monkeypatch.setattr(prompt_utils_module, "PROMPT_DIR", tmp_path)

    assert load_prompt("  assistant_trim_prompt.md  ") == "trim ok"


def test_load_prompt_raises_on_empty_name():
    with pytest.raises(ValueError):
        load_prompt("   ")


def test_load_prompt_raises_when_missing_md_suffix(monkeypatch, tmp_path):
    monkeypatch.setattr(prompt_utils_module, "PROMPT_DIR", tmp_path)

    with pytest.raises(ValueError):
        load_prompt("assistant_demo_prompt")


def test_load_prompt_raises_when_suffix_is_not_md(monkeypatch, tmp_path):
    monkeypatch.setattr(prompt_utils_module, "PROMPT_DIR", tmp_path)

    with pytest.raises(ValueError):
        load_prompt("assistant_demo_prompt.txt")


def test_load_prompt_raises_when_file_not_found(monkeypatch, tmp_path):
    monkeypatch.setattr(prompt_utils_module, "PROMPT_DIR", tmp_path)

    with pytest.raises(FileNotFoundError):
        load_prompt("missing_prompt.md")


def test_load_prompt_uses_memory_cache_after_first_success(monkeypatch, tmp_path):
    prompt_file = tmp_path / "assistant_demo_prompt.md"
    prompt_file.write_text("v1", encoding="utf-8")
    monkeypatch.setattr(prompt_utils_module, "PROMPT_DIR", tmp_path)

    assert load_prompt("assistant_demo_prompt.md") == "v1"

    prompt_file.write_text("v2", encoding="utf-8")
    assert load_prompt("assistant_demo_prompt.md") == "v1"


def test_load_prompt_reads_nested_prompt_path(monkeypatch, tmp_path):
    prompt_file = tmp_path / "system" / "order.md"
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    prompt_file.write_text("nested prompt", encoding="utf-8")
    monkeypatch.setattr(prompt_utils_module, "PROMPT_DIR", tmp_path)

    assert load_prompt("system/order.md") == "nested prompt"


def test_load_prompt_rejects_absolute_path(monkeypatch, tmp_path):
    monkeypatch.setattr(prompt_utils_module, "PROMPT_DIR", tmp_path)
    absolute_prompt_path = str((tmp_path / "assistant_demo_prompt.md").resolve())

    with pytest.raises(ValueError):
        load_prompt(absolute_prompt_path)


def test_load_prompt_rejects_parent_traversal(monkeypatch, tmp_path):
    monkeypatch.setattr(prompt_utils_module, "PROMPT_DIR", tmp_path)

    with pytest.raises(ValueError):
        load_prompt("../secret.md")


def test_load_prompt_rejects_symlink_escape(monkeypatch, tmp_path):
    prompt_root = tmp_path / "prompt"
    prompt_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(prompt_utils_module, "PROMPT_DIR", prompt_root)

    outside_file = tmp_path / "outside.md"
    outside_file.write_text("outside", encoding="utf-8")

    symlink_path = prompt_root / "system" / "leak.md"
    symlink_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        symlink_path.symlink_to(outside_file)
    except OSError:
        pytest.skip("当前环境不支持创建符号链接")

    with pytest.raises(ValueError):
        load_prompt("system/leak.md")
