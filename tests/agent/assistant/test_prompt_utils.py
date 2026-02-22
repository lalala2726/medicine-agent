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

    assert load_prompt("assistant_demo_prompt") == "hello prompt"


def test_load_prompt_trims_name_whitespace(monkeypatch, tmp_path):
    prompt_file = tmp_path / "assistant_trim_prompt.md"
    prompt_file.write_text("trim ok", encoding="utf-8")
    monkeypatch.setattr(prompt_utils_module, "PROMPT_DIR", tmp_path)

    assert load_prompt("  assistant_trim_prompt  ") == "trim ok"


def test_load_prompt_raises_on_empty_name():
    with pytest.raises(ValueError):
        load_prompt("   ")


def test_load_prompt_raises_when_file_not_found(monkeypatch, tmp_path):
    monkeypatch.setattr(prompt_utils_module, "PROMPT_DIR", tmp_path)

    with pytest.raises(FileNotFoundError):
        load_prompt("missing_prompt")


def test_load_prompt_uses_memory_cache_after_first_success(monkeypatch, tmp_path):
    prompt_file = tmp_path / "assistant_demo_prompt.md"
    prompt_file.write_text("v1", encoding="utf-8")
    monkeypatch.setattr(prompt_utils_module, "PROMPT_DIR", tmp_path)

    assert load_prompt("assistant_demo_prompt") == "v1"

    prompt_file.write_text("v2", encoding="utf-8")
    assert load_prompt("assistant_demo_prompt") == "v1"


def test_load_prompt_without_and_with_md_share_same_cache_key(monkeypatch, tmp_path):
    prompt_file = tmp_path / "assistant_demo_prompt.md"
    prompt_file.write_text("v1", encoding="utf-8")
    monkeypatch.setattr(prompt_utils_module, "PROMPT_DIR", tmp_path)

    assert load_prompt("assistant_demo_prompt") == "v1"

    prompt_file.write_text("v2", encoding="utf-8")
    assert load_prompt("assistant_demo_prompt.md") == "v1"
