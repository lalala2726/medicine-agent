from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image as PilImage
from pypdf import PdfReader

from app.services.file_loader.base import (
    FileLoader,
    ImageInfo,
    PageContent,
    ensure_image_output_dir,
    save_image_bytes,
)


def _filter_to_mime(filter_value: object) -> Optional[str]:
    """
    将 PDF 图片过滤器名称映射为 MIME 类型。

    Args:
        filter_value: PDF 图片对象的 /Filter 属性值

    Returns:
        对应的 MIME 类型字符串，如果未知则返回 None
    """
    if not filter_value:
        return None
    if isinstance(filter_value, list):
        filter_name = str(filter_value[0])
    else:
        filter_name = str(filter_value)

    # PDF 图片过滤器到 MIME 类型的映射表
    mapping = {
        "/DCTDecode": "image/jpeg",
        "/JPXDecode": "image/jp2",
        "/FlateDecode": "image/png",
        "/CCITTFaxDecode": "image/tiff",
        "/JBIG2Decode": "image/jbig2",
    }
    return mapping.get(filter_name)


def _guess_extension(name: Optional[str], mime_type: Optional[str]) -> str:
    """
    根据文件名或 MIME 类型推断图片扩展名。

    Args:
        name: 图片文件名
        mime_type: MIME 类型字符串

    Returns:
        图片扩展名（带点号，如 .jpg）
    """
    if name:
        suffix = Path(name).suffix
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
        "image/jbig2": ".jb2",
    }
    return mapping.get(mime_type or "", ".img")


def _get_image_size(data: bytes) -> Tuple[Optional[int], Optional[int]]:
    """
    从图片二进制数据中获取宽度和高度。

    Args:
        data: 图片的二进制数据

    Returns:
        (宽度, 高度) 元组，如果无法解析则返回 (None, None)
    """
    try:
        with PilImage.open(BytesIO(data)) as image:
            return image.width, image.height
    except Exception:
        return None, None


def _extract_page_images_from_xobject(page) -> List[ImageInfo]:
    """
    从 PDF 页面的 XObject 资源中提取图片元信息（仅标注，无二进制数据）。
    这是兜底方案，当 pypdf 的 page.images 接口不可用时使用。

    Args:
        page: PDF 页面对象

    Returns:
        图片信息列表（不含二进制数据）
    """
    images: List[ImageInfo] = []
    resources = page.get("/Resources") if page else None
    if not resources:
        return images

    xobject = resources.get("/XObject") if resources else None
    if not xobject:
        return images

    for name in xobject:
        try:
            obj = xobject[name].get_object()
            if obj.get("/Subtype") != "/Image":
                continue
            width = obj.get("/Width")
            height = obj.get("/Height")
            mime_type = _filter_to_mime(obj.get("/Filter"))
            images.append(
                ImageInfo(
                    index=0,  # 临时索引，后续会被更新
                    name=str(name),
                    width=width,
                    height=height,
                    mime_type=mime_type,
                )
            )
        except Exception:
            continue
    return images


def _extract_page_images(
        page,
        output_dir: Path,
        file_stem: str,
        page_number: int,
        start_index: int,
) -> Tuple[List[ImageInfo], int]:
    """
    从 PDF 页面中提取图片并保存到输出目录。

    Args:
        page: PDF 页面对象
        output_dir: 图片输出目录
        file_stem: 文件名（不含扩展名）
        page_number: 页码
        start_index: 起始图片索引

    Returns:
        (图片信息列表, 下一个图片索引)
    """
    # 优先使用 pypdf 提供的图片提取接口
    images: List[ImageInfo] = []
    index = start_index
    page_images = getattr(page, "images", None)
    if page_images:
        for image in page_images:
            data = getattr(image, "data", None)
            name = getattr(image, "name", None)
            mime_type = getattr(image, "mime_type", None) or getattr(
                image, "content_type", None
            )
            extension = _guess_extension(name, mime_type)
            image_path = None
            width = getattr(image, "width", None)
            height = getattr(image, "height", None)

            if data:
                # 保存图片到输出目录
                image_path = save_image_bytes(
                    output_dir=output_dir,
                    file_stem=file_stem,
                    page_number=page_number,
                    index=index,
                    data=data,
                    extension=extension,
                )
                # 如果宽高未提供，尝试从图片数据中解析
                if width is None or height is None:
                    width, height = _get_image_size(data)

            images.append(
                ImageInfo(
                    index=index,
                    name=name,
                    width=width,
                    height=height,
                    mime_type=mime_type,
                    path=str(image_path) if image_path else None,
                )
            )
            index += 1
        return images, index

    # 兜底：仅标注图片元信息（无法保存图片）
    fallback_images = _extract_page_images_from_xobject(page)
    for image in fallback_images:
        image.index = index
        images.append(image)
        index += 1
    return images, index

    # 兜底：仅标注图片元信息（无法保存图片）
    fallback_images = _extract_page_images_from_xobject(page)
    for image in fallback_images:
        image.index = index
        images.append(image)
        index += 1
    return images, index


class PdfLoader(FileLoader):
    """PDF 解析器，按页提取文本和图片标注。"""

    def parse(
            self, file_path: Path, output_dir: Optional[Path] = None
    ) -> List[PageContent]:
        """
        解析 PDF 文件：按页提取文本和图片。

        Args:
            file_path: PDF 文件路径
            output_dir: 图片输出目录（可选）

        Returns:
            按页组织的内容列表
        """
        reader = PdfReader(str(file_path))
        pages: List[PageContent] = []
        image_output_dir = ensure_image_output_dir(output_dir)
        image_index = 1
        for index, page in enumerate(reader.pages, start=1):
            # 每页分别提取文本和图片
            text = page.extract_text() or ""
            images, image_index = _extract_page_images(
                page=page,
                output_dir=image_output_dir,
                file_stem=file_path.stem,
                page_number=index,
                start_index=image_index,
            )
            pages.append(PageContent(page_number=index, text=text, images=images))
        return pages
