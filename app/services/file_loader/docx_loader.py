from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.core.exceptions import ServiceException
from app.services.file_loader.base import FileLoader, ImageInfo, PageContent


def _iter_block_items(document: Document) -> Iterable[Paragraph | Table]:
    """按文档顺序遍历段落和表格。"""
    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield Table(child, document)


def _block_text(block: Paragraph | Table) -> str:
    # 统一抽取段落/表格文本
    if isinstance(block, Paragraph):
        return block.text.strip()

    rows: List[str] = []
    for row in block.rows:
        values: List[str] = []
        for cell in row.cells:
            cell_text = " ".join(cell.text.split())
            values.append(cell_text)
        if any(values):
            rows.append("\t".join(values))
    return "\n".join(rows)


def _block_has_page_break(paragraph: Paragraph) -> bool:
    # 检测分页符，粗略划分页面
    element = paragraph._p
    for br in element.findall(f".//{qn('w:br')}"):
        if br.get(qn("w:type")) == "page":
            return True
    if element.findall(f".//{qn('w:lastRenderedPageBreak')}"):
        return True
    return False


def _count_images_in_block(block: Paragraph | Table) -> int:
    # 统计段落/表格中的图片数量
    element = block._element
    drawings = element.findall(f".//{qn('w:drawing')}")
    picts = element.findall(f".//{qn('w:pict')}")
    return len(drawings) + len(picts)


def _finalize_pages(pages: List[PageContent]) -> List[PageContent]:
    for page in pages:
        page.text = page.text.strip()
    return pages


def _parse_docx(file_path: Path) -> List[PageContent]:
    document = Document(str(file_path))
    pages: List[PageContent] = [PageContent(page_number=1, text="")]
    image_index = 1

    for block in _iter_block_items(document):
        text = _block_text(block)
        if text:
            pages[-1].text += text + "\n"

        # 记录当前页内图片
        image_count = _count_images_in_block(block)
        for _ in range(image_count):
            pages[-1].images.append(
                ImageInfo(index=image_index, name=f"image_{image_index}")
            )
            image_index += 1

        if isinstance(block, Paragraph) and _block_has_page_break(block):
            pages.append(PageContent(page_number=pages[-1].page_number + 1, text=""))

    return _finalize_pages(pages)


def _parse_with_unstructured(file_path: Path) -> List[PageContent]:
    try:
        from unstructured.partition.auto import partition
    except Exception as exc:
        raise ServiceException("缺少 unstructured 依赖，无法解析该格式") from exc

    try:
        elements = partition(filename=str(file_path))
    except Exception as exc:
        raise ServiceException(f"解析文件失败: {exc}") from exc

    pages_map: Dict[int, PageContent] = {}
    for element in elements:
        # unstructured 的 page_number 可能缺失，默认归为第 1 页
        page_number = getattr(element.metadata, "page_number", None) or 1
        text = element.text or ""
        page = pages_map.setdefault(page_number, PageContent(page_number=page_number, text=""))
        if text:
            if page.text:
                page.text += "\n"
            page.text += text

    pages = [pages_map[key] for key in sorted(pages_map)]
    return _finalize_pages(pages)


class WordLoader(FileLoader):
    """Word 解析器，支持 docx（优先）及 doc（降级解析）。"""

    def parse(self, file_path: Path) -> List[PageContent]:
        suffix = file_path.suffix.lower()
        if suffix == ".docx":
            return _parse_docx(file_path)
        if suffix == ".doc":
            return _parse_with_unstructured(file_path)
        raise ServiceException(f"不支持的 Word 格式: {suffix}")
