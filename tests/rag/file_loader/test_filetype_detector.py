from pathlib import Path

import pytest

from app.core.exception.exceptions import ServiceException
from app.rag.file_loader.detectors import filetype_detector
from app.rag.file_loader.types import FileKind


def test_detect_file_kind_prefers_mime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    测试目的：验证二次识别优先使用 MIME 映射结果，而不是文件后缀。
    预期结果：返回 MIME 对应的 FileKind。
    """
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello", encoding="utf-8")
    monkeypatch.setattr(
        filetype_detector,
        "_detect_mime_type",
        lambda _: "application/pdf",
    )

    file_kind, mime_type = filetype_detector.detect_file_kind(file_path)

    assert file_kind == FileKind.PDF
    assert mime_type == "application/pdf"


def test_detect_file_kind_fallbacks_to_extension_when_mime_unknown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    测试目的：验证 MIME 无法识别时，会回退到文件后缀映射。
    预期结果：返回后缀映射得到的 FileKind。
    """
    file_path = tmp_path / "note.md"
    file_path.write_text("# title", encoding="utf-8")
    monkeypatch.setattr(filetype_detector, "_detect_mime_type", lambda _: None)

    file_kind, mime_type = filetype_detector.detect_file_kind(file_path)

    assert file_kind == FileKind.MARKDOWN
    assert mime_type is None


def test_detect_file_kind_raises_when_unknown_type(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    测试目的：验证 MIME 和后缀都无法识别时会抛出异常。
    预期结果：抛出 ServiceException。
    """
    file_path = tmp_path / "blob.unknown"
    file_path.write_bytes(b"\x00\x01\x02")
    monkeypatch.setattr(filetype_detector, "_detect_mime_type", lambda _: None)

    with pytest.raises(ServiceException):
        filetype_detector.detect_file_kind(file_path)
