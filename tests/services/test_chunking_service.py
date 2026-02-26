import importlib.util

import app.rag.chunking.token_splitter as token_splitter_module
from app.rag.chunking import ChunkStrategyType, SplitConfig, split_file


def test_length_chunker_splits_by_size(tmp_path):
    text = "A" * 120
    file_path = tmp_path / "sample.txt"
    file_path.write_text(text, encoding="utf-8")

    chunks = split_file(
        file_path,
        ChunkStrategyType.LENGTH,
        SplitConfig(chunk_size=50, chunk_overlap=0),
    )

    assert len(chunks) >= 3
    assert all(chunk.text for chunk in chunks)


def test_title_chunker_splits_by_headers(tmp_path):
    content = "# 标题一\n内容1\n## 标题二\n内容2\n### 标题三\n内容3"
    file_path = tmp_path / "sample.md"
    file_path.write_text(content, encoding="utf-8")

    chunks = split_file(file_path, ChunkStrategyType.TITLE)

    assert len(chunks) >= 2
    assert any("标题一" in chunk.text or "内容1" in chunk.text for chunk in chunks)


def test_token_chunker_requires_tiktoken(tmp_path, monkeypatch):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello world", encoding="utf-8")

    if importlib.util.find_spec("tiktoken") is None:
        try:
            split_file(file_path, ChunkStrategyType.TOKEN)
        except Exception as exc:  # ServiceException
            assert "tiktoken" in str(exc)
    else:
        class _FakeTokenTextSplitter:
            def __init__(self, **_kwargs):
                pass

            def split_text(self, text: str):
                return [text]

        monkeypatch.setattr(token_splitter_module, "TokenTextSplitter", _FakeTokenTextSplitter)
        chunks = split_file(
            file_path, ChunkStrategyType.TOKEN, SplitConfig(chunk_size=2, chunk_overlap=0)
        )
        assert len(chunks) >= 1
