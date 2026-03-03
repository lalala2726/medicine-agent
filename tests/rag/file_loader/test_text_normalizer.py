from app.rag.file_loader.normalizers.text_normalizer import normalize_page_text
from app.rag.file_loader.types import FileKind


def test_normalize_page_text_general_collapses_spaces_and_blank_lines() -> None:
    """
    测试目的：验证通用文本清洗会压缩行内空格并限制连续空行数量。
    预期结果：冗余空格被折叠，连续空行最多保留两行。
    """
    raw_text = "a    b\r\n\r\n\r\n\r\nc\t\t d   "
    normalized = normalize_page_text(raw_text, FileKind.PDF)
    assert normalized == "a b\n\n\nc d"


def test_normalize_page_text_markdown_preserves_code_block() -> None:
    """
    测试目的：验证 Markdown 清洗不会破坏代码块中的缩进与空格结构。
    预期结果：代码块内容保持原样，非代码块噪声被清理。
    """
    raw_text = "# 标题\r\n\r\n\r\n```python\r\nprint(  1)\r\n```\r\n\r\n"
    normalized = normalize_page_text(raw_text, FileKind.MARKDOWN)
    assert normalized == "# 标题\n\n\n```python\nprint(  1)\n```"


def test_normalize_page_text_excel_keeps_tab_separator() -> None:
    """
    测试目的：验证 Excel 清洗会保留制表符列分隔并压缩单元格内部空格。
    预期结果：列分隔仍是 `\\t`，单元格噪声空白被压缩。
    """
    raw_text = "  姓名   \t  年龄  \n\n\n 张三  \t  18  "
    normalized = normalize_page_text(raw_text, FileKind.EXCEL)
    assert normalized == "姓名\t年龄\n\n\n张三\t18"
