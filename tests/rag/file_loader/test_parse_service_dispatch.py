from pathlib import Path

import pytest

from app.rag.file_loader import service as file_loader_service
from app.rag.file_loader.types import FileKind, ParseOptions, ParsedPage


class _DummyParser:
    """
    功能描述:
        用于单测的假解析器，返回固定页面内容以验证 service 分发逻辑。

    参数说明:
        无。页面结果在 `parse` 方法内固定构造。

    返回值:
        无。

    异常说明:
        无。
    """

    def parse(self, _file_path: Path) -> list[ParsedPage]:
        """
        功能描述:
            返回固定页面列表，便于断言 service 侧清洗和过滤行为。

        参数说明:
            _file_path (Path): 占位参数，不参与逻辑。

        返回值:
            list[ParsedPage]: 固定页面列表。

        异常说明:
            无。
        """
        return [
            ParsedPage(page_number=1, text="a   b"),
            ParsedPage(page_number=2, text="   "),
        ]


def test_parse_downloaded_file_dispatches_by_detected_kind(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    测试目的：验证下载后解析入口按检测到的真实类型分发解析器。
    预期结果：`get_parser` 接收到 detect_file_kind 返回的 FileKind。
    """
    file_path = tmp_path / "demo.txt"
    file_path.write_text("x", encoding="utf-8")
    called: dict[str, FileKind] = {}

    monkeypatch.setattr(file_loader_service, "validate_url_extension", lambda _: ".pdf")
    monkeypatch.setattr(
        file_loader_service,
        "file_kind_from_extension",
        lambda _: FileKind.PDF,
    )
    monkeypatch.setattr(
        file_loader_service,
        "detect_file_kind",
        lambda _: (FileKind.WORD, "application/msword"),
    )

    def _fake_get_parser(file_kind: FileKind):
        called["kind"] = file_kind
        return _DummyParser()

    monkeypatch.setattr(file_loader_service, "get_parser", _fake_get_parser)

    result = file_loader_service.parse_downloaded_file(
        file_path=file_path,
        source_url="https://example.com/report.pdf",
    )

    assert called["kind"] == FileKind.WORD
    assert result.file_kind == FileKind.WORD
    assert result.mime_type == "application/msword"
    assert len(result.pages) == 1
    assert result.pages[0].text == "a b"


def test_parse_downloaded_file_keeps_empty_pages_when_option_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    测试目的：验证关闭空白页过滤选项时，解析入口会保留清洗后的空白页。
    预期结果：返回结果中包含空白页。
    """
    file_path = tmp_path / "demo.txt"
    file_path.write_text("x", encoding="utf-8")

    monkeypatch.setattr(file_loader_service, "validate_url_extension", lambda _: ".txt")
    monkeypatch.setattr(
        file_loader_service,
        "file_kind_from_extension",
        lambda _: FileKind.TEXT,
    )
    monkeypatch.setattr(
        file_loader_service,
        "detect_file_kind",
        lambda _: (FileKind.TEXT, "text/plain"),
    )
    monkeypatch.setattr(file_loader_service, "get_parser", lambda _: _DummyParser())

    result = file_loader_service.parse_downloaded_file(
        file_path=file_path,
        source_url="https://example.com/report.txt",
        options=ParseOptions(drop_empty_pages=False),
    )

    assert len(result.pages) == 2
