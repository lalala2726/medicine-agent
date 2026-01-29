from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image as PilImage
from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.core.exceptions import ServiceException
from app.core.file_loader.base import (
    FileLoader,
    ImageInfo,
    PageContent,
    ensure_image_output_dir,
    save_image_bytes,
)


def _iter_block_items(document: Document) -> Iterable[Paragraph | Table]:
    """
    按文档顺序遍历段落和表格。

    Args:
        document: Word 文档对象

    Yields:
        段落或表格对象
    """
    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield Table(child, document)


def _block_text(block: Paragraph | Table) -> str:
    """
    统一抽取段落/表格文本内容。

    Args:
        block: 段落或表格对象

    Returns:
        提取的文本内容
    """
    if isinstance(block, Paragraph):
        return block.text.strip()

    # 处理表格：将每行单元格用制表符分隔，行间用换行符分隔
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
    """
    检测段落是否包含分页符，用于粗略划分页面。

    Args:
        paragraph: 段落对象

    Returns:
        如果包含分页符返回 True，否则返回 False
    """
    element = paragraph._p
    # 检查显式分页符
    for br in element.findall(f".//{qn('w:br')}"):
        if br.get(qn("w:type")) == "page":
            return True
    # 检查 Word 渲染时的分页符标记
    if element.findall(f".//{qn('w:lastRenderedPageBreak')}"):
        return True
    return False


def _guess_extension(filename: Optional[str], content_type: Optional[str]) -> str:
    """
    根据文件名或 MIME 类型推断图片扩展名。

    Args:
        filename: 图片文件名
        content_type: MIME 类型字符串

    Returns:
        图片扩展名（带点号）
    """
    if filename:
        suffix = Path(filename).suffix
        if suffix:
            return suffix
    # MIME 类型到扩展名的映射表
    mapping = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
        "image/webp": ".webp",
        "image/jp2": ".jp2",
    }
    return mapping.get(content_type or "", ".img")


def _get_image_size(data: bytes) -> Tuple[Optional[int], Optional[int]]:
    """
    从图片二进制数据中获取宽度和高度。

    Args:
        data: 图片二进制数据

    Returns:
        (宽度, 高度) 元组，解析失败返回 (None, None)
    """
    try:
        with PilImage.open(BytesIO(data)) as image:
            return image.width, image.height
    except Exception:
        return None, None


def _extract_images_from_block(
        document: Document,
        block: Paragraph | Table,
        output_dir: Path,
        file_stem: str,
        page_number: int,
        start_index: int,
) -> Tuple[List[ImageInfo], int]:
    """
    解析当前 block（段落或表格）内的嵌入图片并保存到目录。

    Args:
        document: Word 文档对象
        block: 段落或表格对象
        output_dir: 图片输出目录
        file_stem: 文件名（不含扩展名）
        page_number: 页码
        start_index: 起始图片索引

    Returns:
        (图片信息列表, 下一个图片索引)
    """
    images: List[ImageInfo] = []
    index = start_index
    element = block._element

    # 定义 XML 命名空间
    namespaces = {
        "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }
    try:
        # 查找所有 blip 元素（图片引用）
        blips = element.xpath(".//a:blip", namespaces=namespaces)
    except Exception:
        blips = []

    for blip in blips:
        # 获取图片关系 ID
        r_id = blip.get(qn("r:embed"))
        if not r_id:
            continue
        # 从文档关系中获取图片资源
        image_part = document.part.related_parts.get(r_id)
        if not image_part:
            continue
        data = getattr(image_part, "blob", None)
        if not data:
            continue
        content_type = getattr(image_part, "content_type", None)
        partname = getattr(image_part, "partname", None)
        filename = Path(str(partname)).name if partname else None
        extension = _guess_extension(filename, content_type)
        # 保存图片到输出目录
        image_path = save_image_bytes(
            output_dir=output_dir,
            file_stem=file_stem,
            page_number=page_number,
            index=index,
            data=data,
            extension=extension,
        )
        width, height = _get_image_size(data)
        images.append(
            ImageInfo(
                index=index,
                name=filename,
                width=width,
                height=height,
                mime_type=content_type,
                path=str(image_path),
            )
        )
        index += 1
    return images, index


def _finalize_pages(pages: List[PageContent]) -> List[PageContent]:
    """
    清理页面内容：去除多余空白。

    Args:
        pages: 页面内容列表

    Returns:
        清理后的页面内容列表
    """
    for page in pages:
        page.text = page.text.strip()
    return pages


def _parse_docx(file_path: Path, output_dir: Optional[Path]) -> List[PageContent]:
    """
    解析 .docx 文件：按分页符划分页面，提取文本和图片。

    Args:
        file_path: .docx 文件路径
        output_dir: 图片输出目录（可选）

    Returns:
        按页组织的内容列表
    """
    document = Document(str(file_path))
    pages: List[PageContent] = [PageContent(page_number=1, text="")]
    image_output_dir = ensure_image_output_dir(output_dir) if output_dir else None
    image_index = 1

    for block in _iter_block_items(document):
        # 提取文本内容
        text = _block_text(block)
        if text:
            pages[-1].text += text + "\n"

        # 解析并保存当前 block 内的图片
        if image_output_dir:
            images, image_index = _extract_images_from_block(
                document=document,
                block=block,
                output_dir=image_output_dir,
                file_stem=file_path.stem,
                page_number=pages[-1].page_number,
                start_index=image_index,
            )
            if images:
                pages[-1].images.extend(images)

        # 检测分页符，创建新页面
        if isinstance(block, Paragraph) and _block_has_page_break(block):
            pages.append(PageContent(page_number=pages[-1].page_number + 1, text=""))

    return _finalize_pages(pages)


def _parse_with_unstructured(file_path: Path) -> List[PageContent]:
    """
    使用 unstructured 库降级解析旧格式 .doc 文件。

    Args:
        file_path: .doc 文件路径

    Returns:
        按页组织的内容列表

    Raises:
        ServiceException: 缺少依赖或解析失败
    """
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
        page = pages_map.setdefault(
            page_number, PageContent(page_number=page_number, text="")
        )
        if text:
            if page.text:
                page.text += "\n"
            page.text += text

    # 按页码排序
    pages = [pages_map[key] for key in sorted(pages_map)]
    return _finalize_pages(pages)


class WordLoader(FileLoader):
    """Word 解析器，支持 docx（优先）及 doc（降级解析）。"""

    def parse(
            self, file_path: Path, output_dir: Optional[Path] = None
    ) -> List[PageContent]:
        """
        解析 Word 文件。

        Args:
            file_path: Word 文件路径
            output_dir: 图片输出目录（可选）

        Returns:
            按页组织的内容列表

        Raises:
            ServiceException: 不支持的文件格式
        """
        suffix = file_path.suffix.lower()
        if suffix == ".docx":
            return _parse_docx(file_path, output_dir)
        if suffix == ".doc":
            return _parse_with_unstructured(file_path)
        raise ServiceException(f"不支持的 Word 格式: {suffix}")
